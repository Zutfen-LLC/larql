# MoE expert transport: shape comparison and recommendation

**Status:** SPIKE — analysis framework + measurement plan. On-device numbers
pending a CUDA-client host (see *Pending measurement*).
**Task:** GPU-3001
**Base branch:** `feat/cuda-native-activation-kernels`
**Depends on:** GPU-2001 (CudaBackend local-attn + remote-FFN override),
GPU-2004 (local-CUDA-attn + remote-FFN e2e validation), GPU-1003 (pipeline
benchmark harness with transfer-volume metrics).

---

## 1. The three transport shapes

All three serve the same MoE expert block (gate/up + activation + down,
router-weighted sum) for a hybrid-MoE model whose attention + router run on a
GPU client and whose expert weights live on RAM-only shard servers. They differ
only in **call amortization** — how many round trips per decode token.

Wire-format and dispatch references (all verified against the code on this
branch):

| Shape | Client entry | Server route | Wire codec |
|-------|--------------|--------------|------------|
| HTTP / unary layer-batch | `RemoteMoeBackend::forward_moe` → `Shard::call_layer_batch` | `POST /v1/experts/layer-batch[-f16]` (`routes/expert/layer_batch.rs`) | `larql_inference::ffn::moe_remote::wire::{encode,decode}_layer_batch_request` |
| gRPC `ExpertStream` | `RemoteMoeBackend::forward_moe_stream` over `ShardStream` | `ExpertService::expert_stream` (`grpc_expert.rs`, bidirectional) | `larql_router_protocol::ExpertLayerInput` / `ExpertLayerOutput` |
| multi-layer-batch | `RemoteMoeBackend::forward_moe_predispatch` → `Shard::call_multi_layer_batch[_q8k]` | `POST /v1/experts/multi-layer-batch[-q8k]` (`routes/expert/multi_layer_batch.rs`) | `moe_remote::multi_layer_wire::{encode,decode}_multi_layer_request` |

A fourth shape — `POST /v1/expert/{layer}/{expert_id}` (`single.rs`, one expert
per call) and its legacy batch cousin `POST /v1/expert/batch`
(`batch_legacy.rs`) — is the historical baseline. It ships one residual per
expert per call and is **strictly dominated** by layer-batch for the MoE
forward path; it is retained only for single-expert probes. It is not a
candidate default and is not measured here.

### 1.1 Call amortization (derived from the dispatch code)

Notation: `L` = MoE layers (Mixtral 8×7B: 32), `K` = top-k experts per layer
(Mixtral: 2), `S` = shard servers, `active(L)` = shards that own at least one
of the layer's K routed experts.

**HTTP / layer-batch** (`forward_moe`, `backend.rs:97`): routes locally, groups
the K selected experts by owning shard, issues **one `call_layer_batch` per
active shard per layer** (`backend.rs:150–167`). Per decode token:

```
remote_calls/token = Σ_layer active(layer)   ∈ [L, L·S]
```

For the common 1-shard-owns-all deployment: **L = 32 calls/token**. Each call is
a fresh HTTP/1.1 request (TCP/TLS or UDS handshake amortized by the connection
pool, but full HTTP request/response cycle per call).

**gRPC `ExpertStream`** (`forward_moe_stream`, `backend.rs:303`): opens one
persistent bidirectional stream per shard for the whole decode step
(`open_streams`), sends one `ExpertLayerInput` frame per layer, receives one
`ExpertLayerOutput`. The HTTP/2 connection is established once; per-layer cost is
one frame send + one frame recv, no per-frame handshake. Per decode token the
*connection setup* is amortized across all L layers, so effectively:

```
remote_calls/token ≈ S   (1 persistent stream/shard; L frames ride it)
```

The server-side handler (`grpc_expert.rs:148`) processes each frame with the same
`run_experts_cpu_batch` as the HTTP path, so **per-expert compute is identical**
to layer-batch. The win is purely the eliminated per-layer connection/request
overhead (~12 ms HTTP RTT vs ~0.5 ms frame dispatch, per the comment in
`backend.rs:299`).

**multi-layer-batch** (`forward_moe_predispatch`, `backend.rs` tail;
route `multi_layer_batch.rs`): packs **all L layers into one request per shard**,
runs them in parallel via `rayon::par_iter` over layers (`multi_layer_batch.rs:60`).
Per decode token:

```
remote_calls/token = S   (1 request/shard carries all L layers)
```

The `-q8k` variant (`handle_experts_multi_layer_batch_q8k`) has the client
pre-apply `pre_experts_norm` and pre-quantise `h_norm` to Q8K, so the upload
carries `Q8KActivation { qs, d, sums }` instead of an f32/f16 residual — roughly
**4× smaller upload** than f32 at no accuracy cost (Q8K is the native expert
input format).

### 1.2 Payload bytes/token (Mixtral 8×7B: hidden=4096, K=2, L=32)

Residual sizes: f32 = 4096×4 = 16 KiB; f16 = 8 KiB; Q8K-prenormed ≈ 4 KiB
(qs = hidden bytes + d f32 + sums f32).

| Shape | Upload/token | Download/token | Total/token |
|-------|--------------|----------------|-------------|
| HTTP layer-batch (f32) | L × (16 KiB res + K·8 B) ≈ 512 KiB | L × 16 KiB = 512 KiB | **~1.0 MiB** |
| HTTP layer-batch-f16 | L × 8 KiB = 256 KiB | L × 16 KiB = 512 KiB | ~768 KiB |
| gRPC stream | L × 16 KiB = 512 KiB | L × 16 KiB = 512 KiB | **~1.0 MiB** |
| multi-layer-batch (f32) | L × 16 KiB = 512 KiB | L × 16 KiB = 512 KiB | **~1.0 MiB** |
| multi-layer-batch-q8k | L × 4 KiB = 128 KiB | L × 16 KiB = 512 KiB | **~640 KiB** |

Key observation: **payload bytes/token are approximately equal across f32
shapes** (~1 MiB/token). The residual must be shipped for every layer in all
shapes — each layer's input is a distinct post-attention residual, so
multi-layer-batch cannot deduplicate residuals across layers. The real
differentiators are (a) **round-trip count** and (b) the **q8k upload shrink**.

### 1.3 Why round-trip count dominates on LAN

On a 10 Gb-LAN the wire is not the bottleneck (1 MiB/token ≈ 0.1 ms at 10 Gb/s);
**round-trip latency is**. With a 1 ms LAN RTT:

- HTTP layer-batch: 32 calls × 1 ms = **32 ms/token of pure RTT** → caps decode
  at ~31 tok/s before any compute. This matches the `layer_batch.rs` semaphore
  comment's observed ~180 ms/token unthrottled / ~4 ms/token compute-only.
- gRPC stream: 1 connection + 32 frames; frames pipeline over HTTP/2 so RTT is
  paid once, not 32×. Effectively **~1 ms/token RTT** + L × frame-compute.
- multi-layer-batch: 1 call × 1 ms = **1 ms/token RTT** + parallel compute.

On localhost (UDS or loopback, RTT ≈ 0.02 ms) the RTT term collapses and
**compute parallelism dominates**: all three converge to the rayon compute floor
(~4 ms/token on 8 cores per `layer_batch.rs`). The per-call HTTP overhead
(~50–100 µs request/response cycle) × 32 still makes layer-batch slower than
the packed shapes, but the gap is tens of microseconds, not milliseconds.

---

## 2. Recommendation

### Default: multi-layer-batch-q8k for both regimes

| Regime | Recommended | Rationale |
|--------|-------------|-----------|
| **Localhost / UDS** | multi-layer-batch(-q8k) | RTT negligible; 1 call/token minimises HTTP overhead and lets rayon parallelise all L layers in one `par_iter` (`multi_layer_batch.rs:60`). q8k shrinks upload 4× at zero accuracy cost. |
| **LAN (1–2 ms RTT)** | multi-layer-batch-q8k | Collapses 32 round trips → 1, recovering ~31 ms/token of RTT. q8k upload (128 KiB vs 512 KiB) keeps the single request well under one RTT on 10 Gb. |
| **Interactive single-token streaming** | gRPC `ExpertStream` | If the deployment streams one token at a time to a user and cannot buffer a full layer-batch, the persistent gRPC connection avoids reconnect-per-token while still amortising connection setup across layers. Use only when request-level batching is impossible. |

**Do not default to HTTP layer-batch.** It is strictly dominated by
multi-layer-batch: identical payload, identical server compute
(`run_experts_cpu_batch`), but 32× more round trips. Retain it as a fallback
for single-shard debugging and as the baseline column in the measurement table.

**gRPC is the second choice, not the first**, because: (1) the bidirectional
streaming handler (`grpc_expert.rs:148`) still processes layers sequentially
within a token (one frame in, one frame out) unless the client pipelines fire
then collect — which `forward_moe_stream_fire`/`_collect` does, but only across
layers, not across tokens; (2) multi-layer-batch achieves the same 1-call
amortization with simpler HTTP/1.1 + a stateless server, avoiding the gRPC
health/reconnect surface. Prefer gRPC when the connection persistence itself is
the requirement (long-lived interactive sessions).

---

## 3. Measurement plan (executable once a CUDA client is available)

The GPU-1003 harness (`larql-compute-cuda/benches/pipeline.rs`) measures the
**local** pipeline (prefill / decode / TTFT / htod+dtoh bytes / weight-cache hit
rate) with a CPU baseline column. It does **not** exercise the distributed
transport path. The distributed measurement uses the `larql bench` CLI
(`larql-cli/src/commands/primary/bench/`), which already drives all three
transports via `--moe-shards` and `--moe-dispatch {stream,batch}`.

### 3.1 Topology

- **Client:** 1× RTX 3090 (24 GB), CUDA backend, attention + router local.
- **Servers:** RAM-only shard server(s) (`larql serve --experts …`), 64 GB RAM.
- **Links:** (a) localhost / UDS, (b) 10 Gb-E LAN if available.
- **Model:** Mixtral-class hybrid MoE, 8 experts/layer, top-2, L=32, hidden=4096.

### 3.2 Counters (this slice wires the missing one)

The MoE transport metrics module
(`larql-inference/src/ffn/moe_remote/metrics.rs`) already records `calls`,
`request_bytes`, `response_bytes`, and `active_experts` per shard via
`record_call`, invoked in **all three** transport paths
(`shard/expert_batch.rs`, `shard/layer_batch.rs`, `shard/multi_layer.rs`,
`shard/stream.rs`). `generate_with_remote_moe` calls `metrics::print_summary`
which emits per-token `calls`/`req`/`resp` averages to stderr under
`LARQL_MOE_BYTES=1`.

**Gap closed by this slice:** `remote_moe_runtime.rs` hardcoded
`wire_bytes_per_tok: None`, so payload bytes never reached the structured bench
table / JSON — only stderr. This slice adds `RemoteMoeBackend::transport_totals()`
and wires `wire_bytes_per_tok` from the measured-run delta, so
`payload_bytes/token` appears in the `larql bench` table and
`--output json` envelope. `remote_calls/token` is already emitted to stderr via
`print_summary`; promoting it to a first-class `BenchRow` field is filed as
follow-up GPU-3003 (it cascades through ~10 construction sites and is out of
scope for a spike).

### 3.3 Commands (run on the CUDA client)

```bash
# One server owning all experts (localhost).
larql serve --model <mixtral-vindex> --experts 0-7 --bind 127.0.0.1:8081 &

# (1) HTTP layer-batch (current default forward_moe path)
LARQL_MOE_BYTES=1 larql bench <mixtral-vindex> \
  --moe-shards 0-7=http://127.0.0.1:8081 --moe-dispatch stream \
  -n 50 --warmup 5 --output json --output-file layer-batch.json

# (2) gRPC ExpertStream
larql serve --model <mixtral-vindex> --experts 0-7 --grpc-bind 127.0.0.1:8082 &
LARQL_MOE_BYTES=1 larql bench <mixtral-vindex> \
  --moe-shards 0-7=grpc://127.0.0.1:8082 --moe-dispatch stream \
  -n 50 --warmup 5 --output json --output-file grpc-stream.json

# (3) multi-layer-batch (predispatch; all layers packed)
LARQL_MOE_BYTES=1 larql bench <mixtral-vindex> \
  --moe-shards 0-7=http://127.0.0.1:8081 --moe-dispatch batch \
  --moe-predispatch-iters 1 \
  -n 50 --warmup 5 --output json --output-file multi-layer.json

# CPU-client baseline (no GPU): same three commands with --backends cpu.
```

### 3.4 Result table to populate

| Transport | decode tok/s | remote_calls/token | payload_bytes/token | p99 per-token ms | link |
|-----------|-------------|--------------------|---------------------|------------------|------|
| HTTP layer-batch | _pending_ | 32 (1 shard) | ~1.0 MiB | _pending_ | localhost |
| gRPC stream | _pending_ | 1 (amortized) | ~1.0 MiB | _pending_ | localhost |
| multi-layer-batch | _pending_ | 1 | ~1.0 MiB | _pending_ | localhost |
| multi-layer-batch-q8k | _pending_ | 1 | ~640 KiB | _pending_ | localhost |
| CPU-client baseline | _pending_ | n/a (local) | 0 | _pending_ | n/a |
| _(repeat for 10 Gb-LAN)_ | | | | | |

`decode tok/s`, `p99 ms`, and `payload_bytes/token` come from the bench JSON
(`wire_bytes_per_tok` field, wired by this slice). `remote_calls/token` is read
from the `[moe-bytes] SUMMARY` stderr line (`calls` column) or computed from
topology (§1.1).

---

## 4. Concrete improvement: MoE-expert output cache (filed as GPU-3002)

### Gap

Both existing FFN output caches are **WalkFFN-only** and cannot serve the MoE
expert path:

- **L1 (client)** `larql-vindex/src/cache/l1_cache.rs`: keyed on sorted
  gate-KNN feature IDs (`FfnL1Cache::key`) or a dense i16-quantised residual
  (`FfnL1Cache::residual_key`). Used by `walk_ffn_sparse`, not by `forward_moe`.
- **L2 (server)** `larql-server/src/ffn_l2_cache.rs`: same key scheme
  (`FfnL2Cache::key` from sorted feature IDs), `Arc<Vec<f32>>` values, shared
  across clients. Wired into `routes/walk_ffn/core.rs`, **not** into any
  `/v1/expert*` or gRPC expert handler.

The MoE expert routes (`single.rs`, `layer_batch.rs`, `multi_layer_batch.rs`,
`grpc_expert.rs`) call `run_expert` / `run_experts_cpu_batch` with **no cache
lookup** — every expert call recomputes gate/up + down from scratch.

### Why it matters

During autoregressive decode the same `(layer, expert_id)` is invoked every
token, but with a different residual, so a naive exact-match cache will not hit.
However: (1) the Q8K-prenormed path already quantises the residual to ~256
buckets/dim, and (2) repeated phrases / beam-search candidate tokens produce
near-identical residuals (the L1 `residual_key` quantises to i16 for exactly
this reason, with a measured tolerance for cos≥0.999). A server-side cache keyed
on `(layer, expert_id, i16-quantised-residual-bucket)` would hit on:

- repeated tokens in conversational decode (estimated 10–30% hit rate),
- beam-search where multiple candidates share a prefix residual,
- batch decode of the same prompt across concurrent clients (the L2 cache is
  already cross-client).

A hit eliminates the expert matvec (the dominant server cost, ~4 ms/layer on 8
cores) and, for the caller, the round trip if a client-side L1 is added in
parallel.

### Filed task

**GPU-3002 — Server-side MoE expert output cache.** Extend
`larql-server/src/ffn_l2_cache.rs` with a `(layer, expert_id, residual_key)` map
(reusing the `residual_key` i16-quantisation from `l1_cache.rs`), wire a
get-or-compute into `run_expert` / `run_experts_cpu_batch`, and expose hit-rate
via the existing `/v1/stats` JSON. Acceptance: non-zero hit rate on a
repeated-phrase decode trace; <5% overhead on miss path; correctness parity
test against the uncached path.

---

## 5. Follow-up tasks

| ID | Title | Rationale |
|----|-------|-----------|
| **GPU-3002** | Server-side MoE expert output cache | §4. Cuts server compute on repeated residuals; no cache exists today. |
| **GPU-3003** | Promote `remote_calls_per_token` to a first-class `BenchRow` field + table column | This slice surfaces `payload_bytes/token` (`wire_bytes_per_tok`) but leaves `calls/token` in stderr-only (`print_summary`). A structured field requires touching ~10 `BenchRow` construction sites (row.rs, run.rs, output.rs, all runtime files) — out of scope for a spike. |
| **GPU-3004** | Extend GPU-1003 `pipeline.rs` bench to the distributed transport | Today `pipeline.rs` measures only the local CUDA pipeline. Add a distributed group that boots an in-process shard server and drives all three transports, so `cargo bench -p larql-compute-cuda --bench pipeline` produces the §3.4 table directly. |

---

## 6. Pending measurement (honest status)

On-device numbers for the §3.4 table are **not yet collected**. The runner host
available to this slice (`autocode`) has no NVIDIA GPU (`nvidia-smi` not
present), and the self-hosted CUDA runner provisioned under GPU-1001B was not
reachable for this task. The analysis above (call amortization, payload sizes,
regime recommendation) is derived directly from the dispatch and wire-code on
this branch and does not require execution. The measurement plan in §3 is
executable as-is once a CUDA client + shard server pair is available; the only
code change needed (wiring `wire_bytes_per_tok`) is included in this slice.

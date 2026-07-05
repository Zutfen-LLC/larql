# Remote FFN Runbook — Local CUDA Attention + Remote CPU FFN

**Status:** Procedure (GPU-2004). Validated end-to-end test in
`crates/larql-inference/tests/test_remote_ffn_cuda_e2e.rs`.

This runbook describes how to run a CUDA client that performs attention
locally on an NVIDIA GPU and dispatches the feed-forward network (FFN) to a
remote CPU shard. It is the NVIDIA equivalent of the existing Metal/CPU
remote-MoE integration tests — previously no such run existed for CUDA.

---

## Topology

```
  CUDA client                                CPU FFN server
  ───────────                                ──────────────
  CudaBackend                                larql-server --ffn-only
    attention  (Q4_K, GPU-resident)            FFN  (Q4_K, dequant per layer)
    norms / RoPE (GPU)
    FFN → POST /v1/walk-ffn ──────────────►   per-layer gate/up/down + activation
    ◄────────────── FFN output ───────────   (returns residual delta)
```

The client holds the **attention** slice (`larql slice SRC -o client
--preset client`); the server holds the **FFN** slice (`larql slice SRC -o
server --preset server`). See `slice_cmd.rs:114-169` for the preset
definitions. Both slices are produced from a single Q4_K vindex built with
`larql build --quant q4k`.

### Why `generate_with_remote_ffn`, not `generate_with_remote_moe`

For **dense** Q4_K models (Llama-class, Mistral, Gemma 3), the correct
entrypoint is `generate_with_remote_ffn` + `LayerShardedBackend`. The MoE
path (`generate_with_remote_moe`) requires per-layer router weights that
dense architectures do not have — `build_router` returns `None` for dense
layers, so every layer's FFN callback would contribute a zero vector and
the model would emit garbage.

`generate_with_remote_ffn` marks each pipeline layer
`ffn_is_remote = true` (`patch_pipeline_layers_for_remote_ffn`), so the
`CudaBackend::decode_token_with_moe` override (GPU-2001) fires the callback
at every layer instead of running the local FFN. The callback dispatches
the residual to the server via `LayerShardedBackend::forward`, which issues
one `POST /v1/walk-ffn` per layer.

For **hybrid MoE** models (e.g. Gemma 4 26B-A4B), use
`generate_with_remote_moe` with `--moe-shards` instead — that path routes
expert blocks to remote shards and keeps the dense FFN local.

---

## Prerequisites

1. **CUDA-capable NVIDIA GPU** on the client host (`nvidia-smi` shows a device).
2. **Rust toolchain** with the `cuda` feature compiling. The
   `larql-compute-cuda` crate requires the CUDA toolkit (nvcc + cudart).
3. **A Q4_K vindex** built from a dense Llama-class model. Example:
   ```sh
   larql build meta-llama/Llama-2-7b-hf --quant q4k -o output/llama2-7b-q4k.vindex
   ```
4. **Release `larql-server` binary** on the FFN host:
   ```sh
   cargo build --release -p larql-server
   ```
   (A debug build works but is ~20× slower per token.)

---

## Variant A — Localhost (single machine)

Both the CUDA client and the CPU FFN server run on the same host. This is
the fastest way to validate the pipeline; it isolates correctness from
network latency.

### 1. Slice the vindex

```sh
VINDEX=output/llama2-7b-q4k.vindex

# Client slice: attention + embed + norms + tokenizer
larql slice "$VINDEX" -o output/llama2-7b-client.vindex --preset client

# Server slice: FFN + gate + embed + norms + tokenizer
larql slice "$VINDEX" -o output/llama2-7b-server.vindex --preset server
```

### 2. Start the CPU FFN server

```sh
target/release/larql-server output/llama2-7b-server.vindex \
    --ffn-only --port 8090 --host 127.0.0.1 --no-memcheck
```

Wait for the `Mode: ffn-service (--ffn-only)` log line, then verify:

```sh
curl -s http://127.0.0.1:8090/v1/stats | jq .mode
# → "ffn-service"
```

### 3. Run the CUDA client against the remote FFN

```sh
target/release/larql run output/llama2-7b-client.vindex \
    --ffn http://127.0.0.1:8090 --max-tokens 16 \
    "The capital of France is"
```

Expected output banner:
```
Connecting to remote FFN at http://127.0.0.1:8090…
  Attention:  cuda (local)
  FFN:        remote  (http://127.0.0.1:8090)  dispatch=seq
```

### 4. Run the automated test

The integration test (`test_remote_ffn_cuda_e2e.rs`) automates steps 1-3
and asserts the CUDA+remote-FFN output is token-identical to a CPU-only
reference:

```sh
cargo build --release -p larql-server
LARQL_REQUIRE_CUDA=1 cargo test --features cuda -p larql-inference \
    --test test_remote_ffn_cuda_e2e -- --ignored --nocapture
```

The test spawns its own `larql-server --ffn-only` subprocess on a free
port and tears it down on exit. To point it at an already-running server,
set `LARQL_REMOTE_FFN_URL=http://127.0.0.1:8090`.

On success the test prints:
```
CPU reference: [" Paris", ...]
CUDA+remote-FFN: [" Paris", ...]
OK token-identical to CPU reference (5 tokens)
── remote-FFN transport metrics ──
  remote_calls_per_token:   32  (num_layers=32)
  payload_bytes_per_token:  819200
```

---

## Variant B — 10 GbE LAN (two hosts)

Splits the GPU (attention) and CPU (FFN) across two machines connected by
a 10 GbE link. Use this to measure real network transport cost and to let
a memory-constrained GPU box pair with a high-RAM CPU box for large models
(the topology from ADR-0006: a 31B Q4_K model fits an 8 GB GPU client when
the FFN lives elsewhere).

### Hosts

| Role            | Example host       | Requirements                                   |
|-----------------|--------------------|------------------------------------------------|
| CUDA client     | `gpu-box` (10G NIC)| NVIDIA GPU, CUDA toolkit, client slice         |
| CPU FFN server  | `cpu-box`  (10G NIC)| ≥ model-RAM headroom, release `larql-server`  |

### 1. Build slices once (on either host) and copy

```sh
# On the build host:
VINDEX=output/llama2-7b-q4k.vindex
larql slice "$VINDEX" -o llama2-7b-client.vindex --preset client
larql slice "$VINDEX" -o llama2-7b-server.vindex --preset server

# Ship each slice to its host:
rsync -aP llama2-7b-client.vindex/ gpu-box:~/models/
rsync -aP llama2-7b-server.vindex/ cpu-box:~/models/
```

### 2. Start the FFN server on `cpu-box`

```sh
# on cpu-box:
target/release/larql-server ~/models/llama2-7b-server.vindex \
    --ffn-only --port 8090 --host 0.0.0.0 --no-memcheck
```

Bind to `0.0.0.0` so the client can reach it across the LAN. Confirm from
`gpu-box`:

```sh
curl -s http://cpu-box:8090/v1/stats | jq .mode   # → "ffn-service"
```

### 3. Run the CUDA client on `gpu-box`

```sh
# on gpu-box:
target/release/larql run ~/models/llama2-7b-client.vindex \
    --ffn http://cpu-box:8090 --ffn-timeout-secs 30 \
    --max-tokens 16 \
    "The capital of France is"
```

`--ffn-timeout-secs 30` gives headroom for the first-request server-side
dequant warmup on large layers.

### 4. (Optional) Automated test across the LAN

Set `LARQL_REMOTE_FFN_URL` so the test reuses the remote server instead of
spawning a localhost one:

```sh
# on gpu-box:
LARQL_REQUIRE_CUDA=1 \
LARQL_REMOTE_FFN_URL=http://cpu-box:8090 \
cargo test --features cuda -p larql-inference \
    --test test_remote_ffn_cuda_e2e -- --ignored --nocapture
```

---

## Transport metrics

The CLI and the test both report the per-token transport cost. The dense
remote-FFN path makes **one HTTP call per layer per decode step** (the
residual is sent, the FFN output returned), so:

```
remote_calls_per_token  = num_layers
payload_bytes_per_token ≈ num_layers × hidden_size × 4 (f32) × 2 (req + resp)
```

For Llama-2 7B (32 layers, hidden 4096): ~819 KB/token, ~32 calls/token.

`LARQL_MOE_TIMING=1` and `LARQL_MOE_BYTES=1` enable the per-layer timing
and byte-accounting prints (`moe_call_timed` / `metrics::print_delta`) for
the MoE path. The dense `generate_with_remote_ffn` path surfaces transport
bytes via `LayerShardedBackend::wire_bytes_sent` / `wire_bytes_recv`, which
the test reads after generation completes.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `SKIP: CUDA not available` | `LARQL_REQUIRE_CUDA=1` not set, or no GPU / `cuda` feature off | Set the env var; build with `--features cuda`; confirm `nvidia-smi`. |
| `zero wire bytes` assertion | `decode_token_with_moe` override not firing (GPU-2001 not landed) | Confirm the `CudaBackend` `decode_token_with_moe` impl dispatches to `host_decode_token_with_moe`; check `pipeline.rs:243`. |
| Output diverges from CPU reference | Q4_K dequant mismatch between client attention and server FFN, or wrong slice preset | Re-slice with `--preset client` / `--preset server`; verify both sides load the same base model config. |
| `FFN server did not become healthy` | Server OOM during weight load, or wrong port | Check `/tmp/larql-ffn-server-<port>.log`; raise `--memcheck-headroom-mib` or add RAM. |
| `decode_token_with_moe returned None` | A layer uses an unsupported feature (e.g. PLE) on the CUDA path | PLE architectures (Gemma 4 E-series) are not yet supported by the CUDA MoE override; use a non-PLE dense model. |

---

## See also

- `docs/adr/0006-q4k-remote-ffn.md` — the dense-remote FFN topology decision.
- `slice_cmd.rs` preset definitions (`client`, `server`, `browse`).
- `crates/larql-inference/src/layer_graph/grid/remote_ffn.rs` —
  `generate_with_remote_ffn` implementation.
- `crates/larql-compute-cuda/src/pipeline.rs:243` —
  `host_decode_token_with_moe` (the GPU-2001 override).
- `crates/larql-inference/tests/test_generate_q4k_cpu.rs` — the CPU-only
  reference generator used for parity comparison.

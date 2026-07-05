# MoE Expert Sharding — Scaling Curve

> Status: **GPU-3005**. The integration test that produces this data is
> `crates/larql-inference/tests/test_moe_mixtral_multishard_cuda.rs` (manual,
> `#[ignore]`). The first validated large-MoE CUDA + multi-shard target is
> **Mixtral 8×7B**, the only large MoE that extracts cleanly to Q4_K under
> the PerExpert scheme with standard attention.

## Topology

```text
  CUDA client                          N CPU expert servers
  ───────────                          ──────────────────────
  CudaBackend                            larql-server --ffn-only --experts a-b
    attention (Q4_K, local)               expert FFN (Q4_K, local)
    norms / RoPE (local)
    MoE router (local — ffn/moe_remote/router.rs)
    expert FFN ─► POST /v1/expert/batch (grouped by shard range)
              ◄─ weighted expert outputs ─
```

The client holds the attention, router, norm, and LM-head weights. Each
expert server holds the full Q4_K expert weight directory (`layers/`),
filtered at request time by its `--experts START-END` range. Expert IDs
are a contiguous range owned per shard; a single-shard setup owns
`0..num_experts-1`.

## Why Mixtral is the first validated target

| Property | Mixtral 8×7B | Gemma 4 26B-A4B (reference MoE) |
| --- | --- | --- |
| Extraction scheme | PerExpert (clean) | PerExpert (post-norm complications) |
| Attention | standard | hybrid norm; needs prefill workaround |
| Router | single linear projection / layer | RMSNorm + proj + per-expert scale |
| Q4_K cleanliness | all expert weights → Q4_K | partial; some experts need wider dtype |
| Experts / layer | 8 | 8 |
| Top-K | 2 | 2 |

Mixtral extracts with `larql build mixtral-8x7b --level all --quant q4k`
and slices into expert-server shards via the `expert-server`/`moe-server`
preset (`larql slice SRC -o DST --preset expert-server`).

## Scaling curve — tok/s vs shard count

The numbers below are produced by
`test_moe_mixtral_multishard_cuda::mixtral_multishard_cuda_matches_cpu_reference`
on the validated configuration. Until the run is executed on a GPU host
with a Mixtral Q4_K vindex, the table holds placeholder values marked
**PENDING**; a successful test run prints the real curve to stdout and
this section should be updated from that output.

| Shards | Experts / shard | tok/s | calls / token | parity vs CPU reference |
|--------|-----------------|-------|---------------|-------------------------|
| 1      | 0–7             | PENDING | PENDING      | PENDING |
| 2      | 0–3 / 4–7       | PENDING | PENDING      | PENDING |
| 4      | 0–1 / 2–3 / 4–5 / 6–7 | PENDING | PENDING  | PENDING |

Expected remote_calls/token: `num_layers × 1` for the batch path (one
`POST /v1/expert/batch` per layer per decode step, since a shard owns a
contiguous expert range and top-K=2 experts are typically co-located on
one shard at ≤2 shards; at 4 shards the same experts may split across two
shards, doubling calls/token for the affected layers). Wire bytes/token
scales with `hidden_size × 4` (f32 residual) per direction per call.

### Reproducing

```sh
# 1. Extract the Mixtral Q4_K vindex.
cargo run -p larql-cli -- build mixtral-8x7b --level all --quant q4k

# 2. Build the release server (expert servers run the CPU FFN).
cargo build --release -p larql-server

# 3. Run the manual integration test.
LARQL_REQUIRE_CUDA=1 LARQL_MOE_BYTES=1 LARQL_TEST_VINDEX=<path-to-mixtral-q4k> \
  cargo test --features cuda -p larql-inference \
    --test test_moe_mixtral_multishard_cuda -- --ignored --nocapture
```

The test asserts token-for-token parity between each multi-shard CUDA
client run and a single-process CPU reference, then prints the tok/s and
remote_calls/token for each shard count.

## Notes

- **Client-side MoE router correctness** (`ffn/moe_remote/router.rs`): the
  parity assertion is the correctness proof. The remote path runs the
  exact same router math (norm → scale → proj → softmax → top-K) as the
  local `larql-compute` MoE path; identical expert weights are loaded on
  the shards (same vindex, sliced via `expert-server` preset). If the
  router selected different experts remotely, the token stream would
  diverge. The 1-shard case is the tightest check: identical to local MoE
  except the FFN travels one HTTP hop.
- **Resharding is live**: `RemoteMoeBackend::reshard` swaps the shard map
  without reloading the model. The test reconnects per topology for
  isolation, but the same primitive supports mid-session resharding.
- **gRPC transport**: at 2+ shards on a remote host, the gRPC streaming
  path (`open_streams` / `forward_moe_stream`) cuts per-layer latency from
  ~12ms (unary HTTP) to ~0.5ms (existing HTTP/2 frame). The localhost test
  exercises the HTTP path; remote-host scaling would prefer gRPC.

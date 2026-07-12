#!/usr/bin/env python3
"""ST5 first-token oracle — pinned Transformers CPU float32 eager forward.

Produces a trace directory in the ST5 interchange format (the same format
the LARQL F32 capture writes) so the Rust comparator can diff the two without
knowing which side produced them.

Contract (section 2):
  * device: CPU, weight/compute dtype: torch.float32
  * attention implementation: eager
  * use_cache: False, generation: DISABLED (calls forward(), never generate())
  * output_hidden_states: True, return_dict: True
  * trust_remote_code: False, network access: disabled (local_files_only)

Boundary capture is **non-invasive**: it uses `register_forward_pre_hook` to
observe the residual stream entering each decoder layer, each
`pre_feedforward_layernorm` (whose input is the post-attention residual), each
`per_layer_input_gate` (whose input is the post-FFN residual), the final
`norm` (whose input is the pre-final-norm residual), and `lm_head` (whose
input is the post-final-norm hidden). The model's own attention / FFN / PLE /
norm / lm-head math runs untouched. The only arithmetic the oracle performs
on captured quantities is:

  * `post_ple = post_layer / layer_scalar`  — undoes the model's own
    `hidden_states *= self.layer_scalar` so the residual matches the
    LARQL `on_post_ple` boundary (captured before the scalar multiply).
  * `lm_head_raw = model.lm_head(final_norm[-1])` — re-applies the official
    lm-head Linear to the captured post-norm last token to obtain the raw
    logits before softcap (the same quantity LARQL's `dot_proj` produces).

This is observation through hooks, not a second implementation of Gemma 4.

Run (separate process from the LARQL comparison so both models need not
reside in memory simultaneously):

  LARQL_GEMMA4_ST_DIR=/path/to/google-gemma-4-E2B-it \
  LARQL_GEMMA4_ST_REVISION=9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf \
  LARQL_GEMMA4_ST5_ORACLE_DIR=/path/to/oracle-trace \
  python3 scripts/gemma4_first_token_oracle.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PARITY = Path("bench/baselines/gemma4-e2b-tokenizer-prompt-parity-2026-07-12.json")
REVISION_ENV = "LARQL_GEMMA4_ST_REVISION"
EXPECTED_REVISION = "9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf"
PROMPT_ORDER = ("raw_completion", "chat", "arithmetic", "multiturn")

# Stage names — MUST match the LARQL parity::format constants exactly.
STAGE_EMBEDDING = "embedding"
STAGE_LAYER_INPUT = "layer_input"
STAGE_POST_ATTENTION = "post_attention"
STAGE_POST_FFN = "post_ffn"
STAGE_POST_PLE = "post_ple"
STAGE_POST_LAYER = "post_layer"
STAGE_PRE_FINAL_NORM = "pre_final_norm"
STAGE_FINAL_NORM = "final_norm"
STAGE_LM_HEAD_RAW = "lm_head_raw"
STAGE_FINAL_LOGITS = "final_logits"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_f32(path: Path, values: Any) -> str:
    """Write a 1-D float32 tensor as little-endian bytes; return its SHA-256."""
    import numpy as np

    arr = np.ascontiguousarray(values.detach().cpu().float().numpy(), dtype="<f4")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def layer_tag(layer: int | None) -> str:
    return f"_{layer}" if layer is not None else ""


def tensor_entry(stage: str, layer: int | None, values: Any, trace_dir: Path, prompt_id: str) -> dict[str, Any]:
    rel = f"{prompt_id}/{stage}{layer_tag(layer)}.f32"
    sha = write_f32(trace_dir / rel, values)
    return {
        "stage": stage,
        "layer": layer,
        "shape": [int(values.shape[-1])],
        "dtype": "f32",
        "element_count": int(values.shape[-1]),
        "filename": rel,
        "sha256": sha,
        "not_executed": False,
    }


def verify_revision(root: Path) -> str:
    explicit = os.environ.get(REVISION_ENV)
    revision = explicit or root.name
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision or ""):
        raise ValueError(f"revision must be a 40-char SHA; got {revision!r}")
    revision = revision.lower()
    if revision != EXPECTED_REVISION:
        raise ValueError(f"revision {revision} != pinned {EXPECTED_REVISION}")
    manifest = root / ".cache/huggingface/trees" / f"{revision}.json"
    if not manifest.is_file():
        raise ValueError(f"missing pinned Hugging Face snapshot manifest: {manifest}")
    return revision


def safetensors_sha256(root: Path) -> str:
    st = root / "model.safetensors"
    if not st.is_file():
        raise ValueError(f"missing {st}")
    return sha256_file(st)


def load_prompts(parity_path: Path) -> dict[str, list[int]]:
    data = json.loads(parity_path.read_text(encoding="utf-8"))
    out: dict[str, list[int]] = {}
    for fixture in data["fixtures"]:
        pid = fixture["prompt_id"]
        if pid not in PROMPT_ORDER:
            continue
        ids = fixture["larql_token_ids"]
        out[pid] = [int(t) for t in ids]
    missing = [p for p in PROMPT_ORDER if p not in out]
    if missing:
        raise ValueError(f"parity artifact missing prompts: {missing}")
    return out


def run_oracle(source_root: Path, trace_dir: Path, parity_path: Path) -> dict[str, Any]:
    import safetensors
    import torch
    import transformers

    common = {"local_files_only": True, "trust_remote_code": False}
    revision = verify_revision(source_root)
    st_hash = safetensors_sha256(source_root)
    prompts = load_prompts(parity_path)

    model = transformers.Gemma4ForConditionalGeneration.from_pretrained(
        source_root, dtype=torch.float32, attn_implementation="eager", **common
    )
    model.to("cpu").eval()
    tm = model.model.language_model
    num_layers = len(tm.layers)
    torch.set_num_threads(int(os.environ.get("LARQL_GEMMA4_ST_THREADS", "0")) or torch.get_num_threads())

    # Pre-hook capture buffers (per layer / tail).
    cap: dict[str, Any] = {
        "layer_in": {},
        "post_attn": {},
        "post_ffn": {},
        "norm_in": None,
        "lm_in": None,
    }

    def _li(module, inputs, i):
        cap["layer_in"][i] = inputs[0].detach()

    def _pa(module, inputs, i):
        cap["post_attn"][i] = inputs[0].detach()

    def _pf(module, inputs, i):
        cap["post_ffn"][i] = inputs[0].detach()

    def _normin(module, inputs):
        cap["norm_in"] = inputs[0].detach()

    def _lmin(module, inputs):
        cap["lm_in"] = inputs[0].detach()

    handles = []
    for i, layer in enumerate(tm.layers):
        handles.append(layer.register_forward_pre_hook(lambda md, ip, i=i: _li(md, ip, i)))
        handles.append(
            layer.pre_feedforward_layernorm.register_forward_pre_hook(
                lambda md, ip, i=i: _pa(md, ip, i)
            )
        )
        handles.append(
            layer.per_layer_input_gate.register_forward_pre_hook(
                lambda md, ip, i=i: _pf(md, ip, i)
            )
        )
    handles.append(tm.norm.register_forward_pre_hook(_normin))
    handles.append(model.lm_head.register_forward_pre_hook(_lmin))

    prompt_manifests: dict[str, Any] = {}
    all_finite = True
    captured_shapes: dict[str, list[int]] = {}

    try:
        for pid in PROMPT_ORDER:
            token_ids = prompts[pid]
            ids = torch.tensor([token_ids], dtype=torch.long, device="cpu")
            # Reset capture buffers for this prompt.
            for k in ("layer_in", "post_attn", "post_ffn"):
                cap[k].clear()
            cap["norm_in"] = None
            cap["lm_in"] = None

            with torch.inference_mode():
                output = model(
                    ids,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                )

            # ---- Derive the last-token boundary vectors from captures ----
            tensors: list[dict[str, Any]] = []
            last = -1  # last token index

            embedding = cap["layer_in"][0][:, last, :]
            tensors.append(tensor_entry(STAGE_EMBEDDING, None, embedding, trace_dir, pid))

            for i in range(num_layers):
                layer_in = cap["layer_in"][i][:, last, :]
                tensors.append(tensor_entry(STAGE_LAYER_INPUT, i, layer_in, trace_dir, pid))

                post_attn = cap["post_attn"][i][:, last, :]
                tensors.append(tensor_entry(STAGE_POST_ATTENTION, i, post_attn, trace_dir, pid))

                post_ffn = cap["post_ffn"][i][:, last, :]
                tensors.append(tensor_entry(STAGE_POST_FFN, i, post_ffn, trace_dir, pid))

                # post_layer[i] = next layer's input (i<last) else final-norm input.
                if i < num_layers - 1:
                    post_layer = cap["layer_in"][i + 1][:, last, :]
                else:
                    post_layer = cap["norm_in"][:, last, :]
                tensors.append(tensor_entry(STAGE_POST_LAYER, i, post_layer, trace_dir, pid))

                # post_ple undoes the model's own layer_scalar multiply.
                scalar = float(tm.layers[i].layer_scalar)
                post_ple = post_layer / scalar
                tensors.append(tensor_entry(STAGE_POST_PLE, i, post_ple, trace_dir, pid))

            pre_final_norm = cap["norm_in"][:, last, :]
            tensors.append(tensor_entry(STAGE_PRE_FINAL_NORM, None, pre_final_norm, trace_dir, pid))

            final_norm = cap["lm_in"][:, last, :]
            tensors.append(tensor_entry(STAGE_FINAL_NORM, None, final_norm, trace_dir, pid))

            # lm_head_raw: re-apply the official lm-head Linear to the captured
            # post-norm last token (the model's own module, not a reimplementation).
            with torch.inference_mode():
                lm_head_raw = model.lm_head(final_norm.unsqueeze(0))
            lm_head_raw = lm_head_raw[0, 0, :]
            tensors.append(tensor_entry(STAGE_LM_HEAD_RAW, None, lm_head_raw, trace_dir, pid))

            final_logits = output.logits[:, last, :][0, :]
            tensors.append(tensor_entry(STAGE_FINAL_LOGITS, None, final_logits, trace_dir, pid))

            # Finiteness + shape bookkeeping.
            for t in tensors:
                key = f"{t['stage']}@{t['layer']}"
                if key not in captured_shapes:
                    captured_shapes[key] = list(t["shape"])
            finite = all(
                bool(torch.isfinite(t_val).all().item())
                for t_val in [embedding]
                + [cap["post_attn"][i][:, last, :] for i in range(num_layers)]
                + [final_norm, lm_head_raw, final_logits]
            )
            all_finite &= finite

            prompt_manifests[pid] = {
                "token_ids": token_ids,
                "seq_len": len(token_ids),
                "tensors": tensors,
                "finite": finite,
            }
    finally:
        for handle in handles:
            handle.remove()

    manifest = {
        "schema_version": 1,
        "producer": "transformers-oracle",
        "environment": {
            "device": "cpu",
            "dtype": "torch.float32",
            "attention_implementation": "eager",
            "use_cache": False,
            "generation": "disabled (forward only, never generate())",
            "model_class": type(model).__name__,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": __import__("tokenizers").__version__,
            "safetensors": safetensors.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "trust_remote_code": False,
            "local_files_only": True,
        },
        "model": {
            "repository": "google/gemma-4-E2B-it",
            "revision": revision,
            "safetensors_sha256": st_hash,
            "num_hidden_layers": num_layers,
        },
        "captured_shapes": captured_shapes,
        "all_tensors_finite": bool(all_finite),
        "prompts": prompt_manifests,
    }
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity", type=Path, default=DEFAULT_PARITY)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    source = os.environ.get("LARQL_GEMMA4_ST_DIR")
    if not source:
        print("LARQL_GEMMA4_ST_DIR is required", file=sys.stderr)
        return 2
    source_root = Path(source).expanduser().resolve()
    trace_dir = args.output or Path(
        os.environ.get("LARQL_GEMMA4_ST5_ORACLE_DIR", "bench/baselines/st5-oracle-trace")
    ).expanduser().resolve()

    started = datetime.now(timezone.utc).isoformat()
    try:
        manifest = run_oracle(source_root, trace_dir, args.parity)
    except Exception as error:  # Machine-readable failure artifact.
        failure = {
            "schema_version": 1,
            "producer": "transformers-oracle",
            "status": "FAILED",
            "error": {"message": str(error), "type": type(error).__name__},
            "started_utc": started,
        }
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "manifest.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"oracle failed: {error}", file=sys.stderr)
        return 1

    manifest["started_utc"] = started
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "OK" if manifest.get("all_tensors_finite") else "NON_FINITE"
    (trace_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"oracle trace written to {trace_dir} (finite={manifest['all_tensors_finite']})")
    return 0 if manifest["all_tensors_finite"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

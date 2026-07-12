#!/usr/bin/env python3
"""Run a reproducible, local-only Transformers oracle for Gemma 4 sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPORT = Path("bench/baselines/gemma4-e2b-st-oracle.json")
DEFAULT_INVENTORY = Path("bench/baselines/gemma4-e2b-st-source-inventory.json")
REVISION_ENV = "LARQL_GEMMA4_ST_REVISION"
PROMPTS = (
    {
        "id": "raw_completion",
        "raw": "The capital of France is",
        "messages": None,
    },
    {
        "id": "chat",
        "raw": None,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "In one sentence, explain why the sky appears blue."},
        ],
    },
    {
        "id": "arithmetic",
        "raw": None,
        "messages": [
            {"role": "system", "content": "You are a careful assistant."},
            {"role": "user", "content": "What is 17 multiplied by 23? Give only the answer."},
        ],
    },
)

REQUIRED_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
)
REPOSITORY_ID = "google/gemma-4-E2B-it"
CANONICAL_URL = "https://huggingface.co/google/gemma-4-E2B-it/tree/main"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_id(path: Path) -> str:
    digest = hashlib.sha1()
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        parts = path.relative_to(root).parts
        if path.is_file() and ".git" not in parts and ".cache" not in parts:
            yield path


def classify_tensor(name: str) -> str:
    if name.startswith("model.vision_tower."):
        return "VISION_EXCLUDED"
    if name.startswith("model.audio_tower."):
        return "AUDIO_EXCLUDED"
    if name.startswith(("model.embed_vision.", "model.embed_audio.")):
        return "MULTIMODAL_PROJECTOR_EXCLUDED"
    if "mtp" in name.lower() or "draft" in name.lower():
        return "MTP_EXCLUDED"
    prefix = "model.language_model."
    if not name.startswith(prefix):
        return "UNKNOWN_NON_DECODER"
    normalized = name.removeprefix(prefix)
    if normalized == "embed_tokens.weight":
        return "TOKEN_EMBEDDING_REQUIRED"
    if normalized == "norm.weight":
        return "FINAL_NORM_REQUIRED"
    if normalized in {
        "embed_tokens_per_layer.weight",
        "per_layer_model_projection.weight",
        "per_layer_projection_norm.weight",
    } or any(
        part in normalized
        for part in (
            ".per_layer_input_gate.",
            ".per_layer_projection.",
            ".post_per_layer_input_norm.",
        )
    ):
        return "PLE_REQUIRED"
    if ".self_attn." in normalized:
        if normalized.endswith(("q_norm.weight", "k_norm.weight")):
            return "QK_NORM_REQUIRED"
        return "ATTENTION_REQUIRED"
    if "layernorm.weight" in normalized:
        return "LAYER_NORM_REQUIRED"
    if ".mlp." in normalized:
        return "FFN_REQUIRED"
    if normalized.endswith(".layer_scalar"):
        return "ATTENTION_REQUIRED"
    if normalized == "lm_head.weight":
        return "SEPARATE_LM_HEAD_REQUIRED"
    return "UNKNOWN_TEXT_DECODER"


def git_revision(root: Path) -> str | None:
    git = root / ".git"
    if git.is_file():
        match = re.match(r"gitdir:\s*(.+)\s*$", git.read_text(encoding="utf-8"))
        if not match:
            return None
        git = (root / match.group(1)).resolve()
    if not git.is_dir() or not (git / "HEAD").is_file():
        return None
    head = (git / "HEAD").read_text(encoding="ascii").strip()
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", head):
        return head.lower()
    if not head.startswith("ref: "):
        return None
    ref = head[5:]
    loose = git / ref
    if loose.is_file():
        value = loose.read_text(encoding="ascii").strip()
        return value.lower() if re.fullmatch(r"[0-9a-fA-F]{40,64}", value) else None
    for packed in (git / "packed-refs", git.parent / "packed-refs"):
        if packed.is_file():
            for line in packed.read_text(encoding="ascii").splitlines():
                fields = line.split(" ", 1)
                if len(fields) == 2 and fields[1] == ref:
                    return fields[0].lower()
    return None


def resolve_revision(root: Path) -> tuple[str | None, str | None]:
    explicit = os.environ.get(REVISION_ENV)
    if explicit:
        return explicit, REVISION_ENV
    revision = git_revision(root)
    if revision:
        return revision, "git"
    # Hugging Face cache snapshots are named by their immutable repository commit.
    if root.parent.name == "snapshots" and re.fullmatch(r"[0-9a-fA-F]{40,64}", root.name):
        return root.name.lower(), "huggingface_snapshot_path"
    return None, None


def verify_huggingface_manifest(root: Path, revision: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("repository revision must be a full 40-character commit SHA")
    manifest_path = root / ".cache" / "huggingface" / "trees" / f"{revision.lower()}.json"
    if not manifest_path.is_file():
        raise ValueError(f"pinned Hugging Face snapshot manifest is missing for revision {revision}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("unsupported Hugging Face snapshot manifest")
    expected = manifest["files"]
    observed = {path.relative_to(root).as_posix(): path for path in source_files(root)}
    if set(expected) != set(observed):
        raise ValueError("local source file set does not match the pinned snapshot manifest")
    for name, metadata in expected.items():
        path = observed[name]
        if path.stat().st_size != metadata.get("size"):
            raise ValueError(f"snapshot size mismatch: {name}")
        wanted_sha256 = metadata.get("lfs_sha256")
        if wanted_sha256:
            if sha256(path) != wanted_sha256:
                raise ValueError(f"snapshot SHA-256 mismatch: {name}")
        elif git_blob_id(path) != metadata.get("blob_id"):
            raise ValueError(f"snapshot Git blob mismatch: {name}")
    return manifest_path.stat().st_mtime_ns


def validate_source(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"LARQL_GEMMA4_ST_DIR is not a directory: {root}")
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    tensors = sorted(root.glob("*.safetensors"))
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        named = sorted(set(index.get("weight_map", {}).values()))
        missing.extend(name for name in named if not (root / name).is_file())
        unexpected = sorted(path.name for path in tensors if path.name not in named)
        if unexpected:
            raise ValueError("unexpected safetensors shards: " + ", ".join(unexpected))
    if not tensors:
        missing.append("*.safetensors")
    if missing:
        raise ValueError("missing required source files: " + ", ".join(missing))
    return tensors


def build_inventory(
    root: Path,
    revision: str | None,
    revision_source: str | None,
    download_timestamp_ns: int,
) -> dict[str, Any]:
    from safetensors import safe_open

    files = []
    safetensors = []
    dtype_counts: Counter[str] = Counter()
    tensor_count = 0
    classification_counts: Counter[str] = Counter()
    safetensors_bytes = 0
    for path in source_files(root):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        files.append({"path": relative, "sha256": sha256(path), "size_bytes": size})
        if path.suffix == ".safetensors":
            safetensors_bytes += size
            tensors = []
            with safe_open(path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
                for name in sorted(handle.keys()):
                    tensor = handle.get_slice(name)
                    dtype = str(tensor.get_dtype())
                    shape = list(tensor.get_shape())
                    classification = classify_tensor(name)
                    tensors.append({
                        "classification": classification,
                        "dtype": dtype,
                        "name": name,
                        "shape": shape,
                        "source_shard": relative,
                    })
                    classification_counts[classification] += 1
                    dtype_counts[dtype] += 1
                    tensor_count += 1
            safetensors.append({"metadata": metadata, "path": relative, "tensors": tensors})
    return {
        "canonical_url": CANONICAL_URL,
        "classification_counts": dict(sorted(classification_counts.items())),
        "download_timestamp_utc": datetime.fromtimestamp(
            download_timestamp_ns / 1_000_000_000, timezone.utc
        ).isoformat(),
        "dtype_histogram": dict(sorted(dtype_counts.items())),
        "files": files,
        "repository_revision": revision,
        "revision_source": revision_source,
        "revision_verification": "huggingface_snapshot_manifest",
        "repository_id": REPOSITORY_ID,
        "root": "${LARQL_GEMMA4_ST_DIR}",
        "safetensors": safetensors,
        "safetensors_bytes": safetensors_bytes,
        "safetensors_shard_count": len(safetensors),
        "schema_version": 2,
        "tensor_count": tensor_count,
        "total_repository_bytes": sum(item["size_bytes"] for item in files),
    }


def finite_tree(torch: Any, value: Any) -> tuple[bool, int]:
    tensors = 0
    stack = [value]
    while stack:
        item = stack.pop()
        if torch.is_tensor(item):
            tensors += 1
            if not bool(torch.isfinite(item).all().item()):
                return False, tensors
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return True, tensors


def load_official(root: Path, torch: Any, transformers: Any) -> tuple[Any, Any, Any, str]:
    common = {"local_files_only": True, "trust_remote_code": False}
    processor = transformers.AutoProcessor.from_pretrained(root, **common)
    tokenizer = transformers.AutoTokenizer.from_pretrained(root, **common)
    config = transformers.AutoConfig.from_pretrained(root, **common)
    architectures = config.architectures or []
    multimodal = any(term in name.lower() for name in architectures for term in ("conditionalgeneration", "imagetext"))
    model_class = transformers.AutoModelForImageTextToText if multimodal else transformers.AutoModelForCausalLM
    dtype_name = os.environ.get("LARQL_GEMMA4_ST_DTYPE", "auto")
    dtype = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
    device = os.environ.get("LARQL_GEMMA4_ST_DEVICE", "cpu")
    model = model_class.from_pretrained(root, dtype=dtype, **common)
    model.to(device).eval()
    return processor, tokenizer, model, model_class.__name__


def render_prompt(processor: Any, tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    template_owner = processor if getattr(processor, "chat_template", None) else tokenizer
    controls: dict[str, Any] = {"requested": True, "supported": True, "value": False}
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        rendered = template_owner.apply_chat_template(messages, **kwargs)
    except TypeError:
        controls["supported"] = False
        kwargs.pop("enable_thinking")
        rendered = template_owner.apply_chat_template(messages, **kwargs)
    return rendered, controls


def run_oracle(root: Path, revision: str, inventory_path: Path) -> dict[str, Any]:
    import safetensors
    import torch
    import transformers

    processor, tokenizer, model, model_class = load_official(root, torch, transformers)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    special_ids = {
        name: getattr(tokenizer, name, None)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    }
    special_tokens = {
        name: getattr(tokenizer, name, None)
        for name in ("bos_token", "eos_token", "pad_token", "unk_token")
    }
    runs = []
    all_healthy = True
    for prompt in PROMPTS:
        if prompt["messages"] is None:
            rendered = prompt["raw"]
            thinking = {"requested": False, "supported": None, "value": False}
        else:
            rendered, thinking = render_prompt(processor, tokenizer, prompt["messages"])
        add_special_tokens = prompt["messages"] is None
        input_ids = tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        ).input_ids.to(device)
        if prompt["messages"] is None and input_ids[0, 0].item() != tokenizer.bos_token_id:
            bos = torch.tensor([[tokenizer.bos_token_id]], device=device)
            input_ids = torch.cat((bos, input_ids), dim=1)
        ids = input_ids[0].tolist()
        generation_config = transformers.GenerationConfig.from_model_config(model.config)
        generation_config.do_sample = False
        generation_config.num_beams = 1
        generation_config.max_new_tokens = 16
        generation_config.min_new_tokens = 16
        generation_config.eos_token_id = None
        generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                assistant_model=None,
                return_dict_in_generate=True,
                output_logits=True,
                output_hidden_states=True,
            )
        generated = output.sequences[0, input_ids.shape[1] :].tolist()
        decoded = tokenizer.decode(generated, skip_special_tokens=False)
        logits_finite, logits_count = finite_tree(torch, output.logits)
        hidden_finite, hidden_count = finite_tree(torch, output.hidden_states)
        ids_in_range = all(0 <= token < len(tokenizer) for token in generated)
        healthy = len(generated) == 16 and logits_count > 0 and hidden_count > 0 and logits_finite and hidden_finite and ids_in_range
        all_healthy &= healthy
        lowered = decoded.lower()
        coherent = {
            "raw_completion": "paris" in lowered,
            "chat": any(term in lowered for term in ("scatter", "rayleigh")),
            "arithmetic": "391" in decoded,
        }[prompt["id"]]
        placeholders = any(token in decoded for token in ("<|image|>", "<|audio|>", "<|video|>"))
        hidden_thinking = "<|channel>thought" in decoded or "<|think|>" in decoded
        quality = "COHERENT" if coherent and not placeholders and not hidden_thinking else "DEGRADED"
        runs.append({
            "bos_placement": [index for index, token in enumerate(ids) if token == tokenizer.bos_token_id],
            "decoded_new_tokens": decoded,
            "generated_pieces": tokenizer.convert_ids_to_tokens(generated),
            "generated_token_ids": generated,
            "hidden_states": {"finite": hidden_finite, "tensor_count": hidden_count},
            "input_pieces": tokenizer.convert_ids_to_tokens(ids),
            "input_token_ids": ids,
            "logits": {"finite": logits_finite, "tensor_count": logits_count},
            "new_token_count": len(generated),
            "output_quality": quality,
            "prompt_id": prompt["id"],
            "source_messages": prompt["messages"],
            "raw_prompt": prompt["raw"],
            "rendered_prompt": rendered,
            "thinking_control": thinking,
            "token_ids_within_vocabulary": ids_in_range,
        })
    output_quality = "COHERENT" if all(run["output_quality"] == "COHERENT" for run in runs) else "DEGRADED"
    return {
        "environment": {
            "device": str(device),
            "dtype": str(dtype),
            "model_class": model_class,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "safetensors": safetensors.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": __import__("tokenizers").__version__,
            "huggingface_hub": __import__("huggingface_hub").__version__,
        },
        "generation": {
            "do_sample": False,
            "max_new_tokens": 16,
            "min_new_tokens": 16,
            "num_beams": 1,
            "speculative_decoding": False,
        },
        "inventory_path": str(inventory_path),
        "repository_revision": revision,
        "runs": runs,
        "schema_version": 2,
        "source_health": "HEALTHY" if all_healthy else "UNHEALTHY",
        "output_quality": output_quality,
        "tokenizer": {
            "add_bos_token": getattr(tokenizer, "add_bos_token", None),
            "add_eos_token": getattr(tokenizer, "add_eos_token", None),
            "class": type(tokenizer).__name__,
            "special_token_ids": special_ids,
            "special_tokens": special_tokens,
            "vocab_size": len(tokenizer),
            "eos_ids": model.config.eos_token_id,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    args = parser.parse_args()
    report: dict[str, Any] = {"schema_version": 1, "source_health": "FAILED"}
    try:
        source = os.environ.get("LARQL_GEMMA4_ST_DIR")
        if not source:
            raise ValueError("LARQL_GEMMA4_ST_DIR is required")
        root = Path(source).expanduser().resolve()
        validate_source(root)
        revision, revision_source = resolve_revision(root)
        if revision is None:
            raise ValueError(f"repository revision unavailable; set {REVISION_ENV}")
        download_timestamp_ns = verify_huggingface_manifest(root, revision)
        inventory = build_inventory(root, revision, revision_source, download_timestamp_ns)
        write_json(args.inventory, inventory)
        report = run_oracle(root, revision, args.inventory)
        report["revision_source"] = revision_source
    except Exception as error:  # A machine-readable failure artifact is part of the contract.
        report["error"] = {"message": str(error), "type": type(error).__name__}
        write_json(args.output, report)
        print(f"oracle failed: {error}", file=sys.stderr)
        return 1
    write_json(args.output, report)
    return 0 if report["source_health"] == "HEALTHY" and report["output_quality"] == "COHERENT" else 2


if __name__ == "__main__":
    raise SystemExit(main())

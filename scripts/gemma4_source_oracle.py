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
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and ".git" not in path.relative_to(root).parts:
            yield path


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


def validate_source(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"LARQL_GEMMA4_ST_DIR is not a directory: {root}")
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    tensors = sorted(root.glob("*.safetensors"))
    if not tensors:
        missing.append("*.safetensors")
    if missing:
        raise ValueError("missing required source files: " + ", ".join(missing))
    return tensors


def build_inventory(root: Path, revision: str | None, revision_source: str | None) -> dict[str, Any]:
    from safetensors import safe_open

    files = []
    safetensors = []
    dtype_counts: Counter[str] = Counter()
    tensor_count = 0
    for path in source_files(root):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        files.append({"path": relative, "sha256": sha256(path), "size_bytes": size})
        if path.suffix == ".safetensors":
            tensors = []
            with safe_open(path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
                for name in sorted(handle.keys()):
                    tensor = handle.get_slice(name)
                    dtype = str(tensor.get_dtype())
                    shape = list(tensor.get_shape())
                    tensors.append({"dtype": dtype, "name": name, "shape": shape})
                    dtype_counts[dtype] += 1
                    tensor_count += 1
            safetensors.append({"metadata": metadata, "path": relative, "tensors": tensors})
    return {
        "dtype_histogram": dict(sorted(dtype_counts.items())),
        "files": files,
        "repository_revision": revision,
        "revision_source": revision_source,
        "root": str(root),
        "safetensors": safetensors,
        "schema_version": 1,
        "tensor_count": tensor_count,
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
    model = model_class.from_pretrained(root, torch_dtype=dtype, **common)
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
        input_ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
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
        logits_finite, logits_count = finite_tree(torch, output.logits)
        hidden_finite, hidden_count = finite_tree(torch, output.hidden_states)
        healthy = len(generated) == 16 and logits_count > 0 and hidden_count > 0 and logits_finite and hidden_finite
        all_healthy &= healthy
        runs.append({
            "decoded_new_tokens": tokenizer.decode(generated, skip_special_tokens=False),
            "generated_pieces": tokenizer.convert_ids_to_tokens(generated),
            "generated_token_ids": generated,
            "hidden_states": {"finite": hidden_finite, "tensor_count": hidden_count},
            "input_pieces": tokenizer.convert_ids_to_tokens(ids),
            "input_token_ids": ids,
            "logits": {"finite": logits_finite, "tensor_count": logits_count},
            "new_token_count": len(generated),
            "prompt_id": prompt["id"],
            "source_messages": prompt["messages"],
            "raw_prompt": prompt["raw"],
            "rendered_prompt": rendered,
            "thinking_control": thinking,
        })
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
        "schema_version": 1,
        "source_health": "HEALTHY" if all_healthy else "UNHEALTHY",
        "tokenizer": {
            "add_bos_token": getattr(tokenizer, "add_bos_token", None),
            "add_eos_token": getattr(tokenizer, "add_eos_token", None),
            "class": type(tokenizer).__name__,
            "special_token_ids": special_ids,
            "special_tokens": special_tokens,
            "vocab_size": len(tokenizer),
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
        inventory = build_inventory(root, revision, revision_source)
        write_json(args.inventory, inventory)
        if revision is None:
            raise ValueError(f"repository revision unavailable; set {REVISION_ENV}")
        report = run_oracle(root, revision, args.inventory)
        report["revision_source"] = revision_source
    except Exception as error:  # A machine-readable failure artifact is part of the contract.
        report["error"] = {"message": str(error), "type": type(error).__name__}
        write_json(args.output, report)
        print(f"oracle failed: {error}", file=sys.stderr)
        return 1
    write_json(args.output, report)
    return 0 if report["source_health"] == "HEALTHY" else 2


if __name__ == "__main__":
    raise SystemExit(main())

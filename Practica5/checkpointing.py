from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from causal_llm import CausalLLM
from ner import NERLLM
from tokenizer import BPETokenizer


def save_checkpoint(
    path: str | Path,
    *,
    task: str,
    model: torch.nn.Module,
    tokenizer: BPETokenizer,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": 1,
        "task": task,
        "model_state_dict": model.state_dict(),
        "tokenizer_state": tokenizer.state_dict(),
        "model_config": model_config,
        "training_config": training_config,
        "metrics": metrics or {},
        "metadata": metadata or {},
    }
    torch.save(payload, output_path)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"El archivo {path} no parece un checkpoint valido.")
    return payload


def build_causal_from_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[CausalLLM, BPETokenizer, dict[str, Any]]:
    payload = load_checkpoint(path, map_location=map_location)
    if payload.get("task") != "causal":
        raise ValueError(
            f"Se esperaba un checkpoint causal y se encontro {payload.get('task')!r}."
        )

    tokenizer = BPETokenizer.from_state_dict(payload["tokenizer_state"])
    model = CausalLLM(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(map_location)
    model.eval()
    return model, tokenizer, payload


def build_ner_from_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[NERLLM, BPETokenizer, dict[str, Any]]:
    payload = load_checkpoint(path, map_location=map_location)
    if payload.get("task") != "ner":
        raise ValueError(
            f"Se esperaba un checkpoint NER y se encontro {payload.get('task')!r}."
        )

    tokenizer = BPETokenizer.from_state_dict(payload["tokenizer_state"])
    model = NERLLM(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(map_location)
    model.eval()
    return model, tokenizer, payload

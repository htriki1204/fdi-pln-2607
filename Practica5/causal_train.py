from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """Ventana deslizante sobre un tensor de tokens para language modeling."""

    def __init__(self, data: torch.Tensor, seq_len: int) -> None:
        if len(data) <= seq_len:
            raise ValueError(
                "El corpus tokenizado es demasiado corto para el contexto pedido."
            )
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _split_data(
    tokens: list[int],
    context_size: int,
    train_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor(tokens, dtype=torch.long)
    min_samples = context_size + 2
    if len(data) < 2 * min_samples:
        return data, data

    split = int(train_ratio * len(data))
    split = max(min_samples, min(split, len(data) - min_samples))
    return data[:split], data[split:]


def make_dataloaders(
    tokens: list[int],
    *,
    context_size: int,
    batch_size: int,
    train_ratio: float,
) -> tuple[DataLoader, DataLoader]:
    train_data, val_data = _split_data(tokens, context_size, train_ratio)
    train_ds = TextDataset(train_data, context_size)
    val_ds = TextDataset(val_data, context_size)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def _run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    total_loss = 0.0
    device = next(model.parameters()).device
    is_train = optimizer is not None
    model.train(is_train)

    context_manager = torch.enable_grad() if is_train else torch.no_grad()
    with context_manager:
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            if optimizer is not None:
                optimizer.zero_grad()

            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("El modelo causal no ha devuelto loss.")

            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def train_causal_model(
    model: torch.nn.Module,
    tokens: list[int],
    *,
    epochs: int,
    context_size: int,
    batch_size: int,
    lr: float,
    train_ratio: float,
    seed: int,
) -> dict[str, object]:
    set_seed(seed)
    train_dl, val_dl = make_dataloaders(
        tokens,
        context_size=context_size,
        batch_size=batch_size,
        train_ratio=train_ratio,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_dl, optimizer=optimizer)
        val_loss = _run_epoch(model, val_dl, optimizer=None)
        history.append(
            {"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss}
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_state_dict": best_state,
    }

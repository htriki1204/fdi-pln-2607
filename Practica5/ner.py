"""Modelo NER, dataset y alineamiento palabra -> BPE.

El flujo de datos:

  palabras anotadas (word-level, BIO)
        |
        |  align_to_bpe()       <- asigna B-/I- a cada sub-token
        v
  sub-tokens + etiquetas BIO
        |
        |  NERDataset + collate_ner
        v
  batches (ids, labels) listos para cross_entropy
        |
        v
  NERLLM (Transformer + cabeza lineal por token)
"""

from __future__ import annotations

import copy
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset

from causal_train import set_seed
from transformer import Transformer

# Etiquetas NER en esquema BIO
LABEL2ID = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def align_to_bpe(words, word_labels, tokenizer):
    """Alinea etiquetas BIO de palabras a los sub-tokens de BPE.

    Como el tokenizador BPE puede partir una palabra en varios sub-tokens,
    hay que decidir que etiqueta dar a cada trozo. Regla: la B- se queda
    en el primer sub-token y los siguientes son I-; asi la cabeza marca
    la frontera de la entidad a nivel palabra.

      palabra 'alice' con etiqueta B-PER, BPE la parte en ['al', 'ice']
         -> al: B-PER, ice: I-PER
      palabra 'cheshire' con I-PER, BPE en ['ch', 'es', 'h', 'ire']
         -> todos I-PER
      palabra 'wonderland' con B-LOC, BPE en ['won', 'der', 'land']
         -> won: B-LOC, der: I-LOC, land: I-LOC
      palabras O -> todos sus sub-tokens O
      espacios entre palabras -> O

    Devuelve (token_ids, token_labels) con etiquetas como strings.
    """
    token_ids = []
    token_labels = []
    space_ids = tokenizer.encode(" ")
    for i, (word, label) in enumerate(zip(words, word_labels)):
        if i > 0:
            token_ids.extend(space_ids)
            token_labels.extend(["O"] * len(space_ids))
        word_ids = tokenizer.encode(word)
        token_ids.extend(word_ids)
        if label.startswith("B-"):
            inside = "I-" + label[2:]
            token_labels.append(label)
            token_labels.extend([inside] * (len(word_ids) - 1))
        else:
            token_labels.extend([label] * len(word_ids))
    return token_ids, token_labels


def explain_alignment(words, word_labels, tokenizer):
    """Imprime el alineamiento palabra -> sub-tokens BPE para una frase."""
    print(f"  frase: {' '.join(words)}")
    for word, label in zip(words, word_labels):
        ids = tokenizer.encode(word)
        pieces = [tokenizer.decode([i]) for i in ids]
        if label.startswith("B-"):
            inside = "I-" + label[2:]
            labs = [label] + [inside] * (len(ids) - 1)
        else:
            labs = [label] * len(ids)
        pairs = "  ".join(f"{p}/{l}" for p, l in zip(pieces, labs))
        print(f"    {word:<15} {label:<6} -> {pairs}")


class NERLLM(Transformer):
    """Transformer con cabeza de clasificación por token para NER."""

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model,
        n_heads,
        n_layers,
        expansion,
        dropout,
        num_labels,
    ):
        super().__init__(
            vocab_size, max_seq_len, d_model, n_heads, n_layers, expansion, dropout
        )
        # El transformer ya tiene una representación suficientemente rica,
        # no tenemos más que proyectarla al espacio de etiquetas
        self.ner_head = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, labels=None, attention_mask=None):
        hidden = super().forward(
            input_ids,
            causal=False,
            attention_mask=attention_mask,
        )
        logits = self.ner_head(hidden)
        loss = None
        if labels is not None:
            # cross_entropy espera logits 2D: para cada elemento, una
            # probabilidad por etiqueta.
            # Aplanamos batch y secuencia y tratamos cada token como una muestra
            # independiente:
            #   logits  (n_batches, n_tokens, num_labels) -> (n_batches*n_tokens, num_labels)
            #   labels  (n_batches, n_tokens)             -> (n_batches*n_tokens,)
            # Las posiciones de padding llevan -100 e ignore_index las descarta.
            flat_logits = logits.flatten(0, 1)
            flat_labels = labels.flatten()
            loss = cross_entropy(flat_logits, flat_labels, ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def predict_entities(self, words, tokenizer):
        """Predice etiquetas BIO sobre una lista de **palabras**.

        Trabaja a nivel palabra: para cada palabra reune las etiquetas
        predichas por sus sub-tokens BPE y elige una representante (la primera
        no-O, si existe). Despues agrupa palabras consecutivas con la misma
        entidad para devolver la entidad completa.

        Si el alineado BPE supera `max_seq_len`, trocea internamente por
        palabras para no superar la ventana de contexto.
        """
        self.eval()
        ids_full, _ = align_to_bpe(words, ["O"] * len(words), tokenizer)
        if len(ids_full) > self.max_seq_len:
            entities = []
            for chunk in _chunk_words(words, tokenizer, self.max_seq_len):
                entities.extend(self.predict_entities(chunk, tokenizer))
            return entities

        # Reconstruimos input_ids junto con el rango de sub-tokens de cada
        # palabra (paralelo a `align_to_bpe`).
        token_ids = []
        word_spans = []
        space_ids = tokenizer.encode(" ")
        for i, word in enumerate(words):
            if i > 0:
                token_ids.extend(space_ids)
            start = len(token_ids)
            token_ids.extend(tokenizer.encode(word))
            word_spans.append((start, len(token_ids)))

        device = next(self.parameters()).device
        input_ids = torch.tensor([token_ids], device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        logits, _ = self(input_ids, attention_mask=attention_mask)
        sub_preds = [ID2LABEL[p] for p in logits.argmax(-1)[0].tolist()]

        # Etiqueta por palabra: preferimos la cabeza si es no-O; si la cabeza
        # es O pero algun sub-token interior predice entidad, tomamos esa.
        # Si hay B- y I- mezclados preferimos B-.
        word_labels = []
        for start, end in word_spans:
            sub_labels = sub_preds[start:end]
            non_o = [lab for lab in sub_labels if lab != "O"]
            if not non_o:
                word_labels.append("O")
                continue
            b_labels = [lab for lab in non_o if lab.startswith("B-")]
            word_labels.append(b_labels[0] if b_labels else non_o[0])

        # Reagrupamos palabras BIO consecutivas en entidades.
        entities = []
        i = 0
        while i < len(words):
            lab = word_labels[i]
            if lab == "O":
                i += 1
                continue
            kind = lab.split("-")[1]
            j = i + 1
            while j < len(words) and word_labels[j] == f"I-{kind}":
                j += 1
            text = " ".join(words[i:j]).strip()
            if text:
                entities.append((text, kind))
            i = j
        return entities


class NERDataset(Dataset):
    """Dataset de NER: aplica `align_to_bpe` a cada frase y convierte a tensores."""

    def __init__(self, ner_data, tokenizer, max_len=128):
        self.samples = []
        for words, labels in ner_data:
            ids, labs = align_to_bpe(words, labels, tokenizer)
            ids = ids[:max_len]
            labs = labs[:max_len]
            self.samples.append(
                (
                    torch.tensor(ids, dtype=torch.long),
                    torch.tensor([LABEL2ID[l] for l in labs], dtype=torch.long),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_ner(batch):
    """Padding al largo maximo del batch y mascara para ignorar padding."""
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded_x = torch.zeros(len(xs), max_len, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(xs), max_len, dtype=torch.bool)
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, : len(x)] = x
        padded_y[i, : len(y)] = y
        attention_mask[i, : len(x)] = True
    return padded_x, padded_y, attention_mask


def load_ner_data(path):
    dataset_path = Path(path)
    if dataset_path.suffix.lower() == ".json":
        raw_samples = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(raw_samples, list):
            raise ValueError(
                f"El JSON NER en {dataset_path} debe ser una lista de muestras."
            )

        sentences = []
        for index, sample in enumerate(raw_samples):
            if not isinstance(sample, dict):
                raise ValueError(
                    f"La muestra {index} de {dataset_path} no es un objeto JSON."
                )

            words = sample.get("words")
            labels = sample.get("labels")
            if not isinstance(words, list) or not isinstance(labels, list):
                raise ValueError(
                    f"La muestra {index} de {dataset_path} debe tener listas 'words' y 'labels'."
                )
            if len(words) != len(labels):
                raise ValueError(
                    f"La muestra {index} de {dataset_path} tiene longitudes incompatibles."
                )

            clean_words = [str(word) for word in words]
            clean_labels = [str(label) for label in labels]
            sentences.append((clean_words, clean_labels))

        if not sentences:
            raise ValueError(f"No se han encontrado muestras NER en {dataset_path}.")
        return sentences

    words = []
    labels = []
    sentences = []

    with open(dataset_path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                if words:
                    sentences.append((words, labels))
                    words, labels = [], []
                continue
            if line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    "Cada linea del dataset NER debe tener al menos palabra y etiqueta."
                )

            words.append(parts[0])
            labels.append(parts[-1])

    if words:
        sentences.append((words, labels))

    if not sentences:
        raise ValueError(f"No se han encontrado muestras NER en {dataset_path}.")

    return sentences


def _split_ner_samples(samples, train_ratio, seed):
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    split = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_ratio)))
    if len(shuffled) == 1:
        return shuffled, shuffled
    return shuffled[:split], shuffled[split:]


def _make_ner_dataloaders(samples, tokenizer, max_len, batch_size, train_ratio, seed):
    train_samples, val_samples = _split_ner_samples(samples, train_ratio, seed)
    train_ds = NERDataset(train_samples, tokenizer, max_len=max_len)
    val_ds = NERDataset(val_samples, tokenizer, max_len=max_len)
    return (
        DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_ner
        ),
        DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_ner
        ),
    )


def _run_ner_epoch(model, dataloader, optimizer=None):
    total_loss = 0.0
    correct = 0
    total = 0
    device = next(model.parameters()).device
    is_train = optimizer is not None
    model.train(is_train)

    context_manager = torch.enable_grad() if is_train else torch.no_grad()
    with context_manager:
        for x, y, attention_mask in dataloader:
            x = x.to(device)
            y = y.to(device)
            attention_mask = attention_mask.to(device)

            if optimizer is not None:
                optimizer.zero_grad()

            logits, loss = model(x, labels=y, attention_mask=attention_mask)
            if loss is None:
                raise RuntimeError("El modelo NER no ha devuelto loss.")

            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            valid = y != -100
            correct += (predictions[valid] == y[valid]).sum().item()
            total += valid.sum().item()

    return total_loss / max(len(dataloader), 1), correct / max(total, 1)


def train_ner_model(
    model,
    samples,
    tokenizer,
    *,
    epochs,
    batch_size,
    lr,
    max_len,
    train_ratio,
    seed,
):
    set_seed(seed)
    train_dl, val_dl = _make_ner_dataloaders(
        samples,
        tokenizer,
        max_len,
        batch_size,
        train_ratio,
        seed,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_ner_epoch(
            model, train_dl, optimizer=optimizer
        )
        val_loss, val_accuracy = _run_ner_epoch(model, val_dl, optimizer=None)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "best_state_dict": best_state,
    }


def _split_words(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", text)


def _chunk_words(words, tokenizer, max_seq_len):
    current = []
    for word in words:
        candidate = current + [word]
        ids, _ = align_to_bpe(candidate, ["O"] * len(candidate), tokenizer)
        if current and len(ids) > max_seq_len:
            yield current
            current = [word]
        else:
            current = candidate
    if current:
        yield current


def extract_entities_from_text(text, model, tokenizer):
    entities = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        words = _split_words(sentence)
        if not words:
            continue
        for chunk in _chunk_words(words, tokenizer, model.max_seq_len):
            entities.extend(model.predict_entities(chunk, tokenizer))
    return entities


def aggregate_entities(entities):
    grouped = defaultdict(lambda: {"text": "", "label": "", "count": 0})
    for text, label in entities:
        key = (text.strip(), label)
        grouped[key]["text"] = key[0]
        grouped[key]["label"] = key[1]
        grouped[key]["count"] += 1

    return sorted(
        grouped.values(),
        key=lambda item: (-item["count"], item["label"], item["text"].lower()),
    )

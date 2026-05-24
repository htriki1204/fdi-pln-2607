from __future__ import annotations

import json
from importlib.resources import files
from itertools import product
from pathlib import Path
from typing import Any

import typer

from causal_llm import CausalLLM
from causal_train import choose_device, set_seed, train_causal_model
from checkpointing import (
    build_causal_from_checkpoint,
    build_ner_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from corpus import load_corpus
from ner import (
    NERLLM,
    NUM_LABELS,
    aggregate_entities,
    extract_entities_from_text,
    load_ner_data,
    train_ner_model,
)
from tokenizer import BPETokenizer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="CLI de la practica 5: entrenamiento causal, generacion y NER.",
)

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(str(files("data")))
DEFAULT_CORPUS = DATA_DIR / "alice_in_wonderland.txt"
DEFAULT_NER_DATASET = DATA_DIR / "ner_dataset.json"


def _parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@app.command("train-causal")
def train_causal(
    corpus_path: Path = typer.Option(DEFAULT_CORPUS, help="Corpus .txt o directorio."),
    output_path: Path = typer.Option(
        Path("artifacts/p5_causal_2607.pth"),
        help="Ruta del checkpoint causal de salida.",
    ),
    vocab_size: int = typer.Option(300, help="Tamaño del vocabulario BPE."),
    context_size: int = typer.Option(128, help="Longitud maxima de contexto."),
    d_model: int = typer.Option(128, help="Dimension interna del transformer."),
    n_heads: int = typer.Option(4, help="Numero de cabezas de atencion."),
    n_layers: int = typer.Option(4, help="Numero de bloques transformer."),
    expansion: int = typer.Option(4, help="Factor de expansion feed-forward."),
    dropout: float = typer.Option(0.1, help="Dropout del modelo."),
    epochs: int = typer.Option(5, help="Numero de epocas."),
    batch_size: int = typer.Option(64, help="Tamano de batch."),
    lr: float = typer.Option(3e-4, help="Learning rate."),
    train_ratio: float = typer.Option(0.9, help="Porcentaje para train."),
    seed: int = typer.Option(42, help="Semilla aleatoria."),
    preview_prompt: str = typer.Option(
        "alice and the rabbit were", help="Prompt de prueba tras entrenar."
    ),
    preview_tokens: int = typer.Option(80, help="Tokens a generar en la prueba."),
) -> None:
    text = load_corpus(corpus_path)
    tokenizer = BPETokenizer(text, vocab_size=vocab_size)
    tokens = tokenizer.encode(text)
    device = choose_device()
    set_seed(seed)

    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "max_seq_len": context_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "expansion": expansion,
        "dropout": dropout,
    }
    training_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "train_ratio": train_ratio,
        "seed": seed,
    }

    model = CausalLLM(**model_config).to(device)
    results = train_causal_model(
        model,
        tokens,
        epochs=epochs,
        context_size=context_size,
        batch_size=batch_size,
        lr=lr,
        train_ratio=train_ratio,
        seed=seed,
    )
    model.load_state_dict(results["best_state_dict"])

    save_checkpoint(
        output_path,
        task="causal",
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        training_config=training_config,
        metrics={
            "best_val_loss": results["best_val_loss"],
            "history": results["history"],
        },
        metadata={"corpus_path": str(corpus_path)},
    )

    preview_ids = tokenizer.encode(preview_prompt)
    generated = model.generate(
        preview_ids,
        max_tokens=preview_tokens,
        temperature=0.8,
    )
    typer.echo(f"Checkpoint guardado en: {output_path}")
    typer.echo(f"Mejor val_loss: {results['best_val_loss']:.4f}")
    typer.echo(f"Texto de prueba: {preview_prompt}{tokenizer.decode(generated)}")


@app.command("generate")
def generate_text(
    weights_path: Path = typer.Option(
        Path("p5_causal_2607.pth"),
        help="Checkpoint causal .pth (default: p5_causal_2607.pth en el cwd).",
    ),
    prompt: str = typer.Option(..., help="Prompt de entrada."),
    max_tokens: int = typer.Option(120, help="Maximo de tokens nuevos."),
    temperature: float = typer.Option(0.8, help="Temperatura de muestreo."),
) -> None:
    device = choose_device()
    model, tokenizer, _ = build_causal_from_checkpoint(
        weights_path, map_location=device
    )
    generated_ids = model.generate(
        tokenizer.encode(prompt),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    typer.echo(prompt + tokenizer.decode(generated_ids))


@app.command("train-ner")
def train_ner(
    dataset_path: Path = typer.Option(
        DEFAULT_NER_DATASET,
        help="Dataset JSON/TSV/CoNLL con palabras y etiquetas.",
    ),
    output_path: Path = typer.Option(
        Path("artifacts/p5_ner_2607.pth"),
        help="Ruta del checkpoint NER de salida.",
    ),
    pretrained_path: Path | None = typer.Option(
        None,
        help="Checkpoint causal o NER previo para inicializar el backbone.",
    ),
    tokenizer_corpus: Path = typer.Option(
        DEFAULT_CORPUS,
        help="Corpus para entrenar tokenizer si no se usa pretrained_path.",
    ),
    vocab_size: int = typer.Option(300, help="Vocabulario BPE si no hay pretrain."),
    max_len: int = typer.Option(128, help="Longitud maxima de secuencia."),
    d_model: int = typer.Option(128, help="Dimension interna si no hay pretrain."),
    n_heads: int = typer.Option(4, help="Numero de cabezas si no hay pretrain."),
    n_layers: int = typer.Option(4, help="Numero de capas si no hay pretrain."),
    expansion: int = typer.Option(4, help="Expansion feed-forward."),
    dropout: float = typer.Option(0.1, help="Dropout del modelo."),
    epochs: int = typer.Option(8, help="Numero de epocas."),
    batch_size: int = typer.Option(16, help="Tamano de batch."),
    lr: float = typer.Option(3e-4, help="Learning rate."),
    train_ratio: float = typer.Option(0.85, help="Porcentaje de train."),
    seed: int = typer.Option(42, help="Semilla aleatoria."),
) -> None:
    device = choose_device()
    set_seed(seed)

    if pretrained_path is not None:
        payload = load_checkpoint(pretrained_path)
        tokenizer = BPETokenizer.from_state_dict(payload["tokenizer_state"])
        base_config = dict(payload["model_config"])
        if max_len > base_config["max_seq_len"]:
            typer.echo(
                "Se ajusta max_len al max_seq_len del checkpoint para mantener "
                "consistencia con los embeddings posicionales."
            )
        model_config = {
            "vocab_size": base_config["vocab_size"],
            "max_seq_len": min(max_len, base_config["max_seq_len"]),
            "d_model": base_config["d_model"],
            "n_heads": base_config["n_heads"],
            "n_layers": base_config["n_layers"],
            "expansion": base_config["expansion"],
            "dropout": dropout,
            "num_labels": NUM_LABELS,
        }
    else:
        corpus_text = load_corpus(tokenizer_corpus)
        tokenizer = BPETokenizer(corpus_text, vocab_size=vocab_size)
        model_config = {
            "vocab_size": tokenizer.vocab_size,
            "max_seq_len": max_len,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "expansion": expansion,
            "dropout": dropout,
            "num_labels": NUM_LABELS,
        }
        payload = None

    samples = load_ner_data(dataset_path)
    model = NERLLM(**model_config).to(device)

    if payload is not None:
        incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            typer.echo(
                "Carga parcial de pesos completada: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    results = train_ner_model(
        model,
        samples,
        tokenizer,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_len=model_config["max_seq_len"],
        train_ratio=train_ratio,
        seed=seed,
    )
    model.load_state_dict(results["best_state_dict"])

    save_checkpoint(
        output_path,
        task="ner",
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        training_config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "train_ratio": train_ratio,
            "seed": seed,
        },
        metrics={
            "best_val_loss": results["best_val_loss"],
            "best_val_accuracy": results["best_val_accuracy"],
            "history": results["history"],
        },
        metadata={
            "dataset_path": str(dataset_path),
            "pretrained_path": str(pretrained_path) if pretrained_path else None,
        },
    )

    typer.echo(f"Checkpoint NER guardado en: {output_path}")
    typer.echo(f"Mejor val_loss: {results['best_val_loss']:.4f}")
    typer.echo(f"Mejor val_accuracy: {results['best_val_accuracy']:.4f}")


@app.command("extract-entities")
def extract_entities(
    weights_path: Path = typer.Option(
        Path("p5_ner_2607.pth"),
        help="Checkpoint NER .pth (default: p5_ner_2607.pth en el cwd).",
    ),
    text_path: Path = typer.Option(..., help="Fichero de texto a analizar."),
) -> None:
    device = choose_device()
    model, tokenizer, _ = build_ner_from_checkpoint(weights_path, map_location=device)
    text = text_path.read_text(encoding="utf-8")
    entities = aggregate_entities(extract_entities_from_text(text, model, tokenizer))

    if not entities:
        typer.echo("No se han encontrado entidades nombradas.")
        return

    for entity in entities:
        typer.echo(
            f"{entity['text']}\t{entity['label']}\tapariciones={entity['count']}"
        )


@app.command("grid-search-causal")
def grid_search_causal(
    corpus_path: Path = typer.Option(DEFAULT_CORPUS, help="Corpus .txt o directorio."),
    output_dir: Path = typer.Option(
        Path("artifacts/grid_causal"),
        help="Directorio donde guardar resultados y mejor checkpoint.",
    ),
    vocab_sizes: str = typer.Option("250,300", help="Lista separada por comas."),
    context_sizes: str = typer.Option("96,128", help="Lista separada por comas."),
    d_models: str = typer.Option("96,128", help="Lista separada por comas."),
    learning_rates: str = typer.Option(
        "0.001,0.0003", help="Lista separada por comas."
    ),
    batch_sizes: str = typer.Option("32,64", help="Lista separada por comas."),
    epoch_values: str = typer.Option("3,5", help="Lista separada por comas."),
    n_heads: int = typer.Option(4, help="Numero de cabezas."),
    n_layer_values: str = typer.Option("2,4", help="Lista separada por comas."),
    expansion: int = typer.Option(4, help="Expansion feed-forward."),
    dropouts: str = typer.Option("0.1,0.2", help="Lista separada por comas."),
    train_ratio: float = typer.Option(0.9, help="Porcentaje de train."),
    seed: int = typer.Option(42, help="Semilla aleatoria."),
) -> None:
    text = load_corpus(corpus_path)
    device = choose_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_val_loss = float("inf")

    for run_id, (
        vocab_size,
        context_size,
        d_model,
        lr,
        batch_size,
        epochs,
        n_layers,
        dropout,
    ) in enumerate(
        product(
            _parse_int_list(vocab_sizes),
            _parse_int_list(context_sizes),
            _parse_int_list(d_models),
            _parse_float_list(learning_rates),
            _parse_int_list(batch_sizes),
            _parse_int_list(epoch_values),
            _parse_int_list(n_layer_values),
            _parse_float_list(dropouts),
        ),
        start=1,
    ):
        set_seed(seed)
        tokenizer = BPETokenizer(text, vocab_size=vocab_size)
        tokens = tokenizer.encode(text)
        model_config = {
            "vocab_size": tokenizer.vocab_size,
            "max_seq_len": context_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "expansion": expansion,
            "dropout": dropout,
        }
        model = CausalLLM(**model_config).to(device)
        training_result = train_causal_model(
            model,
            tokens,
            epochs=epochs,
            context_size=context_size,
            batch_size=batch_size,
            lr=lr,
            train_ratio=train_ratio,
            seed=seed,
        )
        val_loss = float(training_result["best_val_loss"])
        run_result = {
            "run_id": run_id,
            "vocab_size": vocab_size,
            "context_size": context_size,
            "d_model": d_model,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "n_layers": n_layers,
            "dropout": dropout,
            "best_val_loss": val_loss,
        }
        results.append(run_result)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_payload = {
                "model_config": model_config,
                "training_config": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "train_ratio": train_ratio,
                    "seed": seed,
                },
                "tokenizer": tokenizer,
                "state_dict": training_result["best_state_dict"],
                "metrics": {
                    "best_val_loss": val_loss,
                    "history": training_result["history"],
                },
                "selection": run_result,
            }

    if best_payload is None:
        raise typer.BadParameter("No se ha podido ejecutar ninguna configuracion.")

    best_model = CausalLLM(**best_payload["model_config"]).to(device)
    best_model.load_state_dict(best_payload["state_dict"])
    save_checkpoint(
        output_dir / "best_causal_grid.pth",
        task="causal",
        model=best_model,
        tokenizer=best_payload["tokenizer"],
        model_config=best_payload["model_config"],
        training_config=best_payload["training_config"],
        metrics=best_payload["metrics"],
        metadata={
            "selection": best_payload["selection"],
            "corpus_path": str(corpus_path),
        },
    )
    _write_json(
        output_dir / "grid_search_causal.json",
        {"best": best_payload["selection"], "runs": results},
    )
    typer.echo(f"Grid search causal completado. Mejor val_loss: {best_val_loss:.4f}")


@app.command("grid-search-ner")
def grid_search_ner(
    dataset_path: Path = typer.Option(
        DEFAULT_NER_DATASET,
        help="Dataset JSON/TSV/CoNLL con palabras y etiquetas.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/grid_ner"),
        help="Directorio donde guardar resultados y mejor checkpoint.",
    ),
    pretrained_path: Path | None = typer.Option(
        None,
        help="Checkpoint causal o NER previo para inicializar backbone/tokenizer.",
    ),
    tokenizer_corpus: Path = typer.Option(
        DEFAULT_CORPUS,
        help="Corpus para tokenizer si no se usa pretrained_path.",
    ),
    learning_rates: str = typer.Option(
        "0.001,0.0003", help="Lista separada por comas."
    ),
    batch_sizes: str = typer.Option("8,16", help="Lista separada por comas."),
    dropouts: str = typer.Option("0.1,0.2", help="Lista separada por comas."),
    epoch_values: str = typer.Option("6,8", help="Lista separada por comas."),
    max_len: int = typer.Option(128, help="Longitud maxima de secuencia."),
    vocab_size: int = typer.Option(300, help="Vocabulario si no hay pretrain."),
    d_model: int = typer.Option(128, help="Dimension interna si no hay pretrain."),
    n_heads: int = typer.Option(4, help="Numero de cabezas si no hay pretrain."),
    n_layer_values: str = typer.Option(
        "2,4", help="Lista separada por comas si no hay pretrain."
    ),
    expansion: int = typer.Option(4, help="Expansion feed-forward."),
    train_ratio: float = typer.Option(0.85, help="Porcentaje de train."),
    seed: int = typer.Option(42, help="Semilla aleatoria."),
) -> None:
    device = choose_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_ner_data(dataset_path)

    if pretrained_path is not None:
        payload = load_checkpoint(pretrained_path)
        tokenizer = BPETokenizer.from_state_dict(payload["tokenizer_state"])
        base_config = dict(payload["model_config"])
        fixed_config = {
            "vocab_size": base_config["vocab_size"],
            "max_seq_len": min(max_len, base_config["max_seq_len"]),
            "d_model": base_config["d_model"],
            "n_heads": base_config["n_heads"],
            "n_layers": base_config["n_layers"],
            "expansion": base_config["expansion"],
        }
    else:
        payload = None
        corpus_text = load_corpus(tokenizer_corpus)
        tokenizer = BPETokenizer(corpus_text, vocab_size=vocab_size)
        fixed_config = {
            "vocab_size": tokenizer.vocab_size,
            "max_seq_len": max_len,
            "d_model": d_model,
            "n_heads": n_heads,
            "expansion": expansion,
        }

    results: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_val_loss = float("inf")
    layer_values = (
        [fixed_config["n_layers"]]
        if pretrained_path is not None
        else _parse_int_list(n_layer_values)
    )

    for run_id, (lr, batch_size, dropout, epochs, n_layers) in enumerate(
        product(
            _parse_float_list(learning_rates),
            _parse_int_list(batch_sizes),
            _parse_float_list(dropouts),
            _parse_int_list(epoch_values),
            layer_values,
        ),
        start=1,
    ):
        set_seed(seed)
        model_config = {
            **fixed_config,
            "n_layers": n_layers,
            "dropout": dropout,
            "num_labels": NUM_LABELS,
        }
        model = NERLLM(**model_config).to(device)
        if payload is not None:
            model.load_state_dict(payload["model_state_dict"], strict=False)

        training_result = train_ner_model(
            model,
            samples,
            tokenizer,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_len=model_config["max_seq_len"],
            train_ratio=train_ratio,
            seed=seed,
        )
        val_loss = float(training_result["best_val_loss"])
        val_accuracy = float(training_result["best_val_accuracy"])
        run_result = {
            "run_id": run_id,
            "lr": lr,
            "batch_size": batch_size,
            "dropout": dropout,
            "epochs": epochs,
            "n_layers": n_layers,
            "best_val_loss": val_loss,
            "best_val_accuracy": val_accuracy,
        }
        results.append(run_result)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_payload = {
                "model_config": model_config,
                "training_config": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "train_ratio": train_ratio,
                    "seed": seed,
                },
                "state_dict": training_result["best_state_dict"],
                "metrics": {
                    "best_val_loss": val_loss,
                    "best_val_accuracy": val_accuracy,
                    "history": training_result["history"],
                },
                "selection": run_result,
            }

    if best_payload is None:
        raise typer.BadParameter("No se ha podido ejecutar ninguna configuracion.")

    best_model = NERLLM(**best_payload["model_config"]).to(device)
    best_model.load_state_dict(best_payload["state_dict"])
    save_checkpoint(
        output_dir / "best_ner_grid.pth",
        task="ner",
        model=best_model,
        tokenizer=tokenizer,
        model_config=best_payload["model_config"],
        training_config=best_payload["training_config"],
        metrics=best_payload["metrics"],
        metadata={
            "selection": best_payload["selection"],
            "dataset_path": str(dataset_path),
            "pretrained_path": str(pretrained_path) if pretrained_path else None,
        },
    )
    _write_json(
        output_dir / "grid_search_ner.json",
        {"best": best_payload["selection"], "runs": results},
    )
    typer.echo(f"Grid search NER completado. Mejor val_loss: {best_val_loss:.4f}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

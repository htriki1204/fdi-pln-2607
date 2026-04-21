from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mini_llm import MiniLLM
from tokenizer import BPETokenizer


DEFAULT_TEXT_PATH = Path(__file__).with_name("alice_in_wonderland.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini LLM con BPE y atencion causal")
    parser.add_argument(
        "--txt",
        type=Path,
        default=DEFAULT_TEXT_PATH,
        help="Ruta al archivo .txt de entrenamiento",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Numero de pasos de entrenamiento",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Tamano de batch",
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=64,
        help="Longitud maxima de contexto",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Dimension de los embeddings",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="Numero de cabezas de atencion",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Numero de bloques transformer",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=200,
        help="Tamano del vocabulario BPE",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Alice ",
        help="Prompt inicial para generar texto",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Numero de tokens a generar",
    )
    return parser.parse_args()


def leer_texto(ruta_txt: Path) -> str:
    if not ruta_txt.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta_txt}")

    texto = ruta_txt.read_text(encoding="utf-8")
    if not texto.strip():
        raise ValueError(f"El archivo esta vacio: {ruta_txt}")
    return texto


def construir_batch(
    data: torch.Tensor,
    batch_size: int,
    n_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= n_tokens:
        raise ValueError("No hay suficientes tokens para construir un batch.")

    starts = torch.randint(0, len(data) - n_tokens - 1, (batch_size,))
    x = torch.stack([data[i : i + n_tokens] for i in starts]).to(device)
    y = torch.stack([data[i + 1 : i + n_tokens + 1] for i in starts]).to(device)
    return x, y


def entrenar(
    model: MiniLLM,
    data: torch.Tensor,
    batch_size: int,
    n_tokens: int,
    steps: int,
    device: torch.device,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for step in range(1, steps + 1):
        x, y = construir_batch(data, batch_size, n_tokens, device)
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            print(f"Paso {step}/{steps} - loss: {loss.item():.4f}")


def main() -> None:
    args = parse_args()
    texto = leer_texto(args.txt)

    tokenizer = BPETokenizer()
    tokenizer.train(texto, vocab_size=args.vocab_size)

    token_ids = tokenizer.encode(texto, add_special_tokens=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_tokens = min(args.n_tokens, len(token_ids) - 1)
    data = torch.tensor(token_ids, dtype=torch.long)

    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        n_tokens=n_tokens,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dropout=0.1,
    ).to(device)

    x_demo, y_demo = construir_batch(data, batch_size=1, n_tokens=n_tokens, device=device)
    logits, loss = model(x_demo, y_demo)

    print(f"Archivo: {args.txt}")
    print(f"Caracteres: {len(texto)}")
    print(f"Tamano de vocabulario: {tokenizer.vocab_size}")
    print(f"Forma de logits: {tuple(logits.shape)}")
    print(f"Loss inicial: {loss.item():.4f}")
    print(f"Dispositivo: {device}")
    print()

    entrenar(
        model=model,
        data=data,
        batch_size=args.batch_size,
        n_tokens=n_tokens,
        steps=args.steps,
        device=device,
    )

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.eval()
    generated = model.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=0.8)

    print("\nTexto generado de prueba:")
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()

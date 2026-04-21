from __future__ import annotations

from collections import Counter
from typing import Iterable


class BPETokenizer:
    """Byte Pair Encoding Tokenizer entrenado sobre un texto.

    Vocabulario inicial: caracteres unicos del texto. Durante el entrenamiento
    se buscan los pares adyacentes mas frecuentes y se fusionan en nuevos tokens,
    hasta 'vocab_size' veces"""
    def __init__(self, special_tokens: Iterable[str] | None = None) -> None:
        self.special_tokens = list(
            special_tokens or ["<pad>", "<bos>", "<eos>", "<unk>"]
        )
        self.merges: list[tuple[str, str]] = []
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["<unk>"]

    def train(self, text: str, vocab_size: int = 256) -> None:
        if not text:
            raise ValueError("El tokenizer necesita texto para entrenarse.")

        base_tokens = sorted(set(text))
        min_vocab_size = len(self.special_tokens) + len(base_tokens)
        if vocab_size < min_vocab_size:
            raise ValueError(
                f"vocab_size={vocab_size} es demasiado pequeno. "
                f"Necesitas al menos {min_vocab_size}."
            )

        sequence = list(text)
        learned_tokens = set(base_tokens)
        self.merges = []

        while len(self.special_tokens) + len(learned_tokens) < vocab_size:
            pair_counts = Counter(zip(sequence, sequence[1:]))
            if not pair_counts:
                break

            best_pair, frequency = pair_counts.most_common(1)[0]
            if frequency < 2:
                break

            merged_token = "".join(best_pair)
            sequence = self._merge_pair(sequence, best_pair, merged_token)
            self.merges.append(best_pair)
            learned_tokens.add(merged_token)

        vocab = self.special_tokens + sorted(learned_tokens, key=lambda token: (len(token), token))
        self.token_to_id = {token: index for index, token in enumerate(vocab)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self._check_is_trained()

        tokens = list(text)
        for pair in self.merges:
            merged_token = "".join(pair)
            tokens = self._merge_pair(tokens, pair, merged_token)

        token_ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]

        if add_special_tokens:
            return [self.bos_token_id, *token_ids, self.eos_token_id]
        return token_ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        self._check_is_trained()

        pieces: list[str] = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), "<unk>")
            if skip_special_tokens and token in self.special_tokens:
                continue
            pieces.append(token)
        return "".join(pieces)

    def _check_is_trained(self) -> None:
        if not self.token_to_id:
            raise RuntimeError("El tokenizer no esta entrenado todavia.")

    @staticmethod
    def _merge_pair(
        sequence: list[str], pair: tuple[str, str], merged_token: str
    ) -> list[str]:
        merged_sequence: list[str] = []
        index = 0

        while index < len(sequence):
            if (
                index < len(sequence) - 1
                and sequence[index] == pair[0]
                and sequence[index + 1] == pair[1]
            ):
                merged_sequence.append(merged_token)
                index += 2
            else:
                merged_sequence.append(sequence[index])
                index += 1

        return merged_sequence

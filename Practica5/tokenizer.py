# Tokenizador BPE (Byte Pair Encoding) mínimo, entrenado sobre el texto."""
#
# PLN 2025/2026 (FDI UCM)
# Antonio F. G. Sevilla <afgs@ucm.es>


from collections import Counter


class BPETokenizer:
    """Byte Pair Encoding entrenado sobre un texto.

    Vocabulario inicial: caracteres unicos del texto. Durante el
    entrenamiento se buscan los pares adyacentes mas frecuentes y se
    fusionan en nuevos tokens, hasta alcanzar `vocab_size` tokens.

    NOTA: para ser BPE de verdad, tendríamos que hacerlo sobre bytes, no sobre
    caracteres, pero para la práctica funciona bien.
    """

    def __init__(self, text=None, vocab_size=300):
        self.vocab_size = vocab_size
        self.vocab = []
        self.tok2id = {}
        self.merges = []  # lista de ((id_a, id_b), nuevo_id), para encode()

        if text is not None:
            self.train(text, vocab_size=vocab_size)

    def train(self, text, vocab_size=None):
        if not text:
            raise ValueError("El tokenizador necesita texto para entrenarse.")

        self.vocab_size = vocab_size or self.vocab_size
        # Inicializamos con caracteres encontrados en el texto
        self.vocab = sorted(set(text))  # vocab[id] -> token string.
        self.tok2id = {tok: i for i, tok in enumerate(self.vocab)}

        tokens = [self.tok2id[c] for c in text]
        self.merges = []  # lista de ((id_a, id_b), nuevo_id), para encode()

        for new_id in range(len(self.vocab), self.vocab_size):
            pairs = Counter(zip(tokens, tokens[1:]))
            if not pairs:
                break
            best, count = pairs.most_common(1)[0]
            if count < 2:
                break

            new_tok = self.vocab[best[0]] + self.vocab[best[1]]
            self.tok2id[new_tok] = new_id
            self.vocab.append(new_tok)
            self.merges.append((best, new_id))

            tokens = self._apply_merge(tokens, best[0], best[1], new_id)

    @staticmethod
    def _apply_merge(tokens, a, b, new_id):
        """Reemplaza todas las ocurrencias del par (a, b) por new_id."""
        result = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                result.append(new_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    def encode(self, text):
        """Codifica un texto aplicando los merges aprendidos."""
        if not self.vocab:
            raise RuntimeError("El tokenizador no esta entrenado todavia.")
        tokens = [self.tok2id.get(c, 0) for c in text]
        for (a, b), new_id in self.merges:
            tokens = self._apply_merge(tokens, a, b, new_id)
        return tokens

    def decode(self, ids):
        """Decodifica una lista de ids a texto."""
        if not self.vocab:
            raise RuntimeError("El tokenizador no esta entrenado todavia.")
        return "".join(self.vocab[i] for i in ids)

    def state_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "tok2id": self.tok2id,
            "merges": self.merges,
        }

    @classmethod
    def from_state_dict(cls, state):
        tokenizer = cls()
        tokenizer.vocab_size = state["vocab_size"]
        tokenizer.vocab = list(state["vocab"])
        tokenizer.tok2id = {tok: int(idx) for tok, idx in state["tok2id"].items()}
        tokenizer.merges = [
            ((int(a), int(b)), int(new_id)) for (a, b), new_id in state["merges"]
        ]
        return tokenizer

    def __repr__(self):
        pretty = [t.replace("\n", "\\n").replace(" ", "▁") for t in self.vocab]
        joined = "', '".join(pretty)
        return f"{len(self.vocab)} tokens: ['{joined}']"


# Si ejecutamos este módulo directamente, probamos el tokenizador
if __name__ == "__main__":
    import sys
    from pathlib import Path

    files_path = Path(sys.argv[1] if len(sys.argv) > 1 else "resources")
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    textos = "\n\n".join(open(p).read() for p in files_path.glob("*.txt"))
    tokenizer = BPETokenizer(textos, vocab_size=vocab_size)
    print(tokenizer)

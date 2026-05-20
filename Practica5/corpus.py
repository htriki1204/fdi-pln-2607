from pathlib import Path


def load_corpus(path="alice_in_wonderland.txt"):
    corpus_path = Path(path)

    if corpus_path.is_file():
        return corpus_path.read_text(encoding="utf-8")

    if corpus_path.is_dir():
        texts = [
            file_path.read_text(encoding="utf-8")
            for file_path in sorted(corpus_path.glob("*.txt"))
        ]
        if not texts:
            raise FileNotFoundError(f"No .txt files found in directory: {corpus_path}")
        return "\n\n".join(texts)

    raise FileNotFoundError(f"Corpus path not found: {corpus_path}")

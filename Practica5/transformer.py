# Transformer básico comentado
#
# PLN 2025/2026 (FDI UCM)
# Antonio F. G. Sevilla <afgs@ucm.es>

import torch
import torch.nn as nn

from attention import Attention


class FeedForward(nn.Module):
    """Red feed-forward ala perceptron con capa intermedia *más amplia*, y
    activación GELU.

    El factor de expansión permite a la red encontrar y procesar patrones en un
    espacio menos denso que d_model."""

    def __init__(self, d_model, expansion, dropout):
        super().__init__()
        ## Lo llevamos a un espacio mas grande, introducimos no linealidad y despues volvemos a la dimension deseada.
        hidden = expansion * d_model
        # Red fully-connected con una capa oculta
        self.up = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.down = nn.Linear(hidden, d_model)
        # Añadimos dropout para regularizar
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(self.act(self.up(x))))


class Block(nn.Module):
    """Bloque principal del transformer a repetir.

    Incluye las dos cosas principales:
    1. Mecanismo de atención para atender al contexto, aprender matices y ambiguedades.
    2. Red feed-forward, para aprender a abstraer y generar las entradas de la siguiente capa.

    Se incluyen capas de normalización para regularizar el aprendizaje."""

    def __init__(self, max_seq_len, d_model, n_heads, expansion, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, max_seq_len, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, expansion, dropout)

    def forward(self, x, causal=True, attention_mask=None):
        # Aplicamos las capas pero sumando (residuales, skip connections)
        x = x + self.attn(self.norm1(x), causal=causal, attention_mask=attention_mask)
        x = x + self.ff(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Backbone del transformer: embeddings, bloques de atención y normalización final.

    Produce representaciones contextuales (hidden state
    s) para cada token de
    entrada. No realiza ninguna tarea concreta: sirve de base para distintos
    modelos añadiéndole una cabeza específica (generación, clasificación…).

    Parámetros:
      vocab_size   Tamaño del vocabulario
      max_seq_len  Longitud máxima de secuencia
      d_model      Dimensión interna de las representaciones
      n_heads      Número de cabezas de atención
      n_layers     Número de bloques transformer apilados
      expansion    Factor de expansión de la capa feed-forward (típico: 4)
      dropout      Tasa de dropout para regularización
    """

    def __init__(
        self, vocab_size, max_seq_len, d_model, n_heads, n_layers, expansion, dropout
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # El inicio del transformer son los embeddings. Tenemos dos, los de
        # vocabulario y los posicionales (se calculan en forward)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        # Dropout para regularizar el aprendizaje de los embeddings
        self.drop = nn.Dropout(dropout)

        # El corazón del transformer es el bloque principal, con atención y
        # feedforward, que repetimos en secuencia varias veces
        self.blocks = nn.ModuleList(
            [
                Block(max_seq_len, d_model, n_heads, expansion, dropout)
                for _ in range(n_layers)
            ]
        )
        # Una última normalización final
        self.norm = nn.LayerNorm(d_model)

    def forward(self, idx, causal=True, attention_mask=None):
        """Devuelve los hidden states para cada token de idx.

        idx     Tensor (batch, n_tokens) con ids de tokens
        causal  Si True, la atención es causal (solo mira tokens anteriores)
        """
        ## Por ejemplo la frase de hola -> [15,83] Esos serian los indices,
        _, n_tokens = idx.shape  # (batch, numero de tokens). por ejemeplo (2,4) =
        ## [5,9,3,8]
        ## [2.7.1.4]

        if n_tokens > self.max_seq_len:
            raise ValueError(
                f"Secuencia de longitud {n_tokens} mayor que max_seq_len={self.max_seq_len}."
            )

        # Los tokens de vocabulario se entrenan, los posicionales se calculan
        # directamente (en GPU si estamos usando GPU)
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(n_tokens, device=idx.device))
        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x, causal=causal, attention_mask=attention_mask)
        return self.norm(x)  ## (batch_size, n_tokens. d_model)

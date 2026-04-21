from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_tokens: int,
        n_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model debe ser divisible por n_heads.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # La mascara causal impide mirar tokens futuros.
        mask = torch.triu(torch.ones(n_tokens, n_tokens, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attention_scores = attention_scores.masked_fill(causal_mask, float("-inf"))

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = attention_weights @ v
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.proj(context)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_tokens: int,
        n_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attention = Attention(
            d_model=d_model,
            n_tokens=n_tokens,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class MiniLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_tokens: int,
        d_model: int = 128,
        n_heads: int = 1,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.n_tokens = n_tokens

        # Embeddings aprendibles para tokens y posiciones.
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(n_tokens, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_tokens=n_tokens,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self, input_ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.n_tokens:
            raise ValueError(
                f"Longitud {seq_len} mayor que n_tokens={self.n_tokens}."
            )

        positions = torch.arange(seq_len, device=input_ids.device)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions).unsqueeze(0)

        x = self.dropout(token_embeddings + position_embeddings)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(batch_size * seq_len, -1), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self, input_ids: Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> Tensor:
        for _ in range(max_new_tokens):
            input_window = input_ids[:, -self.n_tokens :]
            logits, _ = self(input_window)
            next_token_logits = logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

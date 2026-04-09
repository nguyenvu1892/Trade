"""
transformer.py
--------------
Causal Transformer Encoder cho chuỗi giá XAUUSD.

Kiến trúc:
  Input:  (batch, window_size, n_features)
  → Linear Projection → Positional Encoding → 
  → L × Causal TransformerEncoderLayer (triangular mask) →
  → Global Average Pool →
  → Policy Head (n_actions logits) + Value Head (1 scalar)

Causal mask đảm bảo nến tại thời điểm t chỉ
nhìn thấy nến từ 0 đến t (không rò rỉ tương lai).
"""

import math
import torch
import torch.nn as nn


class XAUTransformer(nn.Module):
    def __init__(
        self,
        n_features:  int   = 10,
        window_size: int   = 128,
        d_model:     int   = 256,
        n_heads:     int   = 8,
        n_layers:    int   = 6,
        dropout:     float = 0.1,
        n_actions:   int   = 3,
    ):
        super().__init__()
        self.window_size = window_size
        self.d_model     = d_model

        # ── Input Projection ──────────────────────────────────────────
        self.input_projection = nn.Linear(n_features, d_model)

        # ── Positional Encoding (sine/cosine, cố định) ────────────────
        self.register_buffer(
            "pos_enc",
            self._build_pos_enc(window_size, d_model)
        )

        # ── Causal Transformer Encoder ────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True,   # Input: (batch, seq, d_model)
            norm_first      = True,   # Pre-LN stabilizes training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Output Heads ──────────────────────────────────────────────
        self.policy_head = nn.Linear(d_model, n_actions)
        self.value_head  = nn.Linear(d_model, 1)

        # ── Causal Mask (triangular) ──────────────────────────────────
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(window_size)
        )

        self._init_weights()

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor shape (batch, window_size, n_features)

        Returns
        -------
        logits : (batch, n_actions)
        value  : (batch, 1)
        """
        # Input projection
        x = self.input_projection(x)                   # (B, W, d_model)

        # Positional encoding
        x = x + self.pos_enc[:x.size(1), :]            # (B, W, d_model)

        # Causal Transformer Encoder
        x = self.encoder(x, mask=self.causal_mask,
                         is_causal=True)                # (B, W, d_model)

        # [CRITICAL FIX] Chỉ lấy token cuối cùng
        # Token W-1 là duy nhất tổng hợp được toàn bộ lịch sử [0..W-1]
        # Mean-pooling sẽ làm loãng thông tin với các token đầu bị mù
        x = x[:, -1, :]                                # (B, d_model)  ← FIX

        # [FIX GRADIENT EARTHQUAKE] Cắt đứt gradient từ Value Head về phần thân Transformer
        # Ở PPO Phase 2, Value Head ngẫu nhiên => Error siêu lớn. Nếu k detach, nó phá nát Core đã train ở BC.
        return self.policy_head(x), self.value_head(x.detach())

    @staticmethod
    def _build_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
        """Tạo Positional Encoding (PE) cố định dạng sine/cosine."""
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe  # (max_len, d_model)

    def _init_weights(self):
        """Xavier init cho Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
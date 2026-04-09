"""
dual_transformer.py
--------------------
Dual-Timeframe Transformer: H1 (macro trend) + M5 (entry timing).

Kiến trúc:
  H1 Encoder (64 bars ≈ 2.5 ngày):  → nhận biết trend lớn
  M5 Encoder (256 bars ≈ 21 giờ):   → xác định thời điểm vào/ra
  Cross-Attention:  M5 query × H1 key/value → kết hợp
  Policy + Value Heads

Input:
  m5_features : (batch, 256, n_features_m5)
  h1_features : (batch, 64,  n_features_h1)

Output:
  logits : (batch, 3)  — Hold/Buy/Sell
  value  : (batch, 1)
"""

import math
import torch
import torch.nn as nn


class DualTimeframeTransformer(nn.Module):
    def __init__(
        self,
        n_features_m5: int = 13,
        n_features_h1: int = 13,
        window_m5:     int = 256,
        window_h1:     int = 64,
        d_model:       int = 256,
        n_heads:       int = 8,
        n_layers_m5:   int = 6,
        n_layers_h1:   int = 3,    # H1 cần ít layers hơn (ít bars hơn)
        n_cross_layers: int = 2,   # Cross-attention layers
        dropout:       float = 0.1,
        n_actions:     int = 3,
    ):
        super().__init__()
        self.window_m5 = window_m5
        self.window_h1 = window_h1
        self.d_model = d_model

        # ── M5 Branch ──────────────────────────────────
        self.m5_projection = nn.Linear(n_features_m5, d_model)
        self.register_buffer("m5_pos_enc", self._build_pos_enc(window_m5, d_model))
        m5_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.m5_encoder = nn.TransformerEncoder(m5_layer, num_layers=n_layers_m5)
        self.register_buffer(
            "m5_causal_mask",
            nn.Transformer.generate_square_subsequent_mask(window_m5)
        )

        # ── H1 Branch ──────────────────────────────────
        self.h1_projection = nn.Linear(n_features_h1, d_model)
        self.register_buffer("h1_pos_enc", self._build_pos_enc(window_h1, d_model))
        h1_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.h1_encoder = nn.TransformerEncoder(h1_layer, num_layers=n_layers_h1)
        self.register_buffer(
            "h1_causal_mask",
            nn.Transformer.generate_square_subsequent_mask(window_h1)
        )

        # ── Cross-Attention: M5 attend to H1 ──────────
        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=d_model, num_heads=n_heads,
                    dropout=dropout, batch_first=True
                ),
                "norm1": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                ),
                "norm2": nn.LayerNorm(d_model),
            })
            for _ in range(n_cross_layers)
        ])

        # ── Fusion ─────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Output Heads ──────────────────────────────
        self.policy_head = nn.Linear(d_model, n_actions)
        self.value_head = nn.Linear(d_model, 1)

        self._init_weights()

    def forward(self, m5_features: torch.Tensor, h1_features: torch.Tensor):
        """
        Parameters
        ----------
        m5_features : (batch, window_m5, n_features_m5)
        h1_features : (batch, window_h1, n_features_h1)

        Returns
        -------
        logits : (batch, n_actions)
        value  : (batch, 1)
        """
        # ── Encode M5 ──
        m5 = self.m5_projection(m5_features)
        m5 = m5 + self.m5_pos_enc[:m5.size(1), :]
        m5 = self.m5_encoder(m5, mask=self.m5_causal_mask, is_causal=True)

        # ── Encode H1 ──
        h1 = self.h1_projection(h1_features)
        h1 = h1 + self.h1_pos_enc[:h1.size(1), :]
        h1 = self.h1_encoder(h1, mask=self.h1_causal_mask, is_causal=True)

        # ── Cross-Attention: M5 attends to H1 context ──
        cross_out = m5
        for layer in self.cross_attention_layers:
            # Pre-norm cross attention
            residual = cross_out
            cross_out = layer["norm1"](cross_out)
            cross_out, _ = layer["cross_attn"](
                query=cross_out,
                key=h1,
                value=h1,
            )
            cross_out = residual + cross_out

            # FFN
            residual = cross_out
            cross_out = layer["norm2"](cross_out)
            cross_out = residual + layer["ffn"](cross_out)

        # ── Lấy token cuối cùng ──
        m5_last = m5[:, -1, :]             # M5 pure context
        cross_last = cross_out[:, -1, :]   # M5 + H1 fused context

        # ── Fusion ──
        combined = self.fusion(torch.cat([m5_last, cross_last], dim=-1))

        # ── Heads ──
        return self.policy_head(combined), self.value_head(combined.detach())

    @staticmethod
    def _build_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

# src/model/tests/test_transformer.py
import torch
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.model.transformer import XAUTransformer


BATCH      = 4
WINDOW     = 64
N_FEATURES = 10
N_ACTIONS  = 3


class TestXAUTransformer:
    def setup_method(self):
        self.model = XAUTransformer(
            n_features = N_FEATURES,
            window_size = WINDOW,
            d_model = 64,
            n_heads = 4,
            n_layers = 2,
            dropout = 0.1,
            n_actions = N_ACTIONS,
        )

    def test_policy_output_shape(self):
        """Policy head phải output (batch, n_actions)."""
        x = torch.randn(BATCH, WINDOW, N_FEATURES)
        logits, value = self.model(x)
        assert logits.shape == (BATCH, N_ACTIONS), (
            f"Policy shape kỳ vọng ({BATCH}, {N_ACTIONS}), nhận {logits.shape}"
        )

    def test_value_output_shape(self):
        """Value head phải output (batch, 1)."""
        x = torch.randn(BATCH, WINDOW, N_FEATURES)
        logits, value = self.model(x)
        assert value.shape == (BATCH, 1), (
            f"Value shape kỳ vọng ({BATCH}, 1), nhận {value.shape}"
        )

    def test_no_nan_in_output(self):
        """Output không được có NaN."""
        x = torch.randn(BATCH, WINDOW, N_FEATURES)
        logits, value = self.model(x)
        assert not torch.isnan(logits).any(), "NaN trong policy logits"
        assert not torch.isnan(value).any(), "NaN trong value output"

    def test_gradient_flows_to_all_layers(self):
        """Gradient phải lưu thông đến tận lớp embedding (layer đầu tiên)."""
        x = torch.randn(BATCH, WINDOW, N_FEATURES)
        logits, value = self.model(x)
        loss = logits.sum() + value.sum()
        loss.backward()

        embedding_weight = self.model.input_projection.weight
        assert embedding_weight.grad is not None, "Gradient không đến được embedding layer"
        assert not torch.isnan(embedding_weight.grad).any(), "NaN trong gradient"

    def test_causal_last_token_only(self):
        """
        [CRITICAL FIX] Phải dùng x[:, -1, :] (token cuối)
        Kiểm tra module bên trong để xem có rò rỉ tương lai không.
        Vì model() chỉ trả về token cuối (điểm hiện tại), nên thay đổi
        bất cứ token nào trong input cũng sẽ làm thay đổi out.
        Do đó test trực tiếp encoder internals:
        """
        x = torch.randn(BATCH, WINDOW, N_FEATURES)
        self.model.eval()
        
        # Test Causal Behavior directly on the encoder
        x_proj = self.model.input_projection(x) + self.model.pos_enc[:x.size(1), :]
        x_mod = x_proj.clone()
        x_mod[:, WINDOW//2:, :] += 999.0  # Thay đổi tương lai
        
        with torch.no_grad():
            out1 = self.model.encoder(x_proj, mask=self.model.causal_mask, is_causal=True)
            out2 = self.model.encoder(x_mod, mask=self.model.causal_mask, is_causal=True)
            
        # Nửa đầu của sequence không được khác nhau (Causality preserved)
        diff = (out1[:, :WINDOW//2, :] - out2[:, :WINDOW//2, :]).abs().max().item()
        assert diff < 1e-4, f"Causal mask bị vi phạm: diff={diff}"

    def test_batch_independence(self):
        """Output của một sample không phụ thuộc sample khác trong cùng batch."""
        x = torch.randn(BATCH, WINDOW, N_FEATURES)
        single = x[0:1]
        self.model.eval()  # [FIX] Tắt dropout để test batch independence
        with torch.no_grad():
            logits_batch, _ = self.model(x)
            logits_single, _ = self.model(single)

        diff = (logits_batch[0] - logits_single[0]).abs().max().item()
        assert diff < 1e-4, (
            f"Batch không độc lập — output sample 0 thay đổi theo batch: {diff}"
        )

    def test_dropout_disabled_in_eval(self):
        """Dropout phải tắt trong eval mode (output deterministic)."""
        x = torch.randn(BATCH, WINDOW, N_FEATURES)
        self.model.eval()
        with torch.no_grad():
            out1, _ = self.model(x)
            out2, _ = self.model(x)
        diff = (out1 - out2).abs().max().item()
        assert diff == 0.0, f"Eval mode không deterministic: diff={diff}"
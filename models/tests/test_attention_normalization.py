import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from models.model import ExternalPerturbAttention
from models.training.checkpoints import load_checkpoint
from models.training.config import TrainingConfig
from models.training.modeling import create_model


class AttentionNormalizationTests(unittest.TestCase):
    def test_projected_vectors_and_ffn_outputs_are_normalized_before_use(self):
        torch.manual_seed(9)
        block = ExternalPerturbAttention(16, 2, dropout=0, legacy_attention_norms=True).eval()
        x = torch.randn(1, 7, 16) * 20
        chem = torch.randn(1, 3, 16) * 30
        attention_inputs = []
        joint_inputs = []
        original = F.scaled_dot_product_attention

        def capture(q, k, v, **kwargs):
            attention_inputs.extend(t.transpose(1, 2).reshape(1, -1, 16).detach()
                                    for t in (q, k, v))
            return original(q, k, v, **kwargs)

        handle = block.joint_norm.register_forward_pre_hook(
            lambda module, args: joint_inputs.append(args[0].detach()))
        with patch("models.model.F.scaled_dot_product_attention", side_effect=capture):
            out = block(x, chem)
        handle.remove()
        vectors = attention_inputs + [joint_inputs[0][..., 16:], out]
        self.assertEqual(len(attention_inputs), 6)
        for value in vectors:
            torch.testing.assert_close(value.mean(-1), torch.zeros_like(value[..., 0]), atol=2e-6, rtol=0)
            torch.testing.assert_close(value.var(-1, unbiased=False), torch.ones_like(value[..., 0]), atol=0.01, rtol=0)

    def test_new_block_has_only_ffn_norms_and_raw_projections(self):
        torch.manual_seed(9)
        block = ExternalPerturbAttention(16, 2, dropout=0).eval()
        self.assertEqual({n for n, m in block.named_modules() if isinstance(m, torch.nn.LayerNorm)},
                         {"chem_ffn_norm", "gene_ffn_norm"})
        x = torch.randn(2, 7, 16, requires_grad=True)
        chem = torch.randn(2, 3, 16, requires_grad=True)
        calls = []
        projections = []
        def forbidden(*args):
            raise AssertionError("Disabled query/joint norm was called")
        block.query_norm.register_forward_pre_hook(forbidden)
        block.joint_norm.register_forward_pre_hook(forbidden)
        original = F.scaled_dot_product_attention
        def capture(q, k, v, **kwargs):
            calls.append((q, k, v))
            return original(q, k, v, **kwargs)
        handle = block.kqv.register_forward_hook(lambda m, a, result: projections.append(result))
        with patch("models.model.F.scaled_dot_product_attention", side_effect=capture):
            out = block(x, chem)
        handle.remove()
        k, v = block.kv_linear(chem).chunk(2, -1)
        for actual, expected in zip(calls[0], (block.q_linear(x), k, v)):
            torch.testing.assert_close(actual, block.split_heads(expected))
        k, q, v = projections[0].chunk(3, -1)
        for actual, expected in zip(calls[1], (q, k, v)):
            torch.testing.assert_close(actual, block.split_heads(expected))
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(chem.grad).all())

    def test_schema_ten_preserves_legacy_norms_and_parameter_order(self):
        cfg = TrainingConfig(dim=16, conc_dim=4, heads=2, layers=1,
                             legacy_attention_norms=True)
        model = create_model(cfg, 7, 5)
        config = cfg.to_dict()
        config.pop("legacy_attention_norms")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            torch.save(dict(schema_version=10, config=config, model=model.state_dict()), path)
            _, state = load_checkpoint(path)
            rebuilt = create_model(TrainingConfig(**state["config"]), 7, 5)
            rebuilt.load_state_dict(state["model"], strict=True)
            self.assertEqual(list(dict(model.named_parameters())), list(dict(rebuilt.named_parameters())))
            self.assertIsInstance(rebuilt.layers[0].chem_q_norm, torch.nn.LayerNorm)

    def test_schema_four_full_model_keeps_original_parameter_keys(self):
        cfg = TrainingConfig(dim=16, conc_dim=4, heads=2, layers=1,
                             attention_normalization=False, chem_ffn_residual=False, legacy_attention_norms=True)
        model = create_model(cfg, 7, 5)
        config = cfg.to_dict()
        config.pop("attention_normalization")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            torch.save(dict(schema_version=4, config=config, model=model.state_dict()), path)
            _, restored = load_checkpoint(path)
            self.assertFalse(restored["config"]["attention_normalization"])
            rebuilt = create_model(TrainingConfig(**restored["config"]), 7, 5)
            rebuilt.load_state_dict(restored["model"], strict=True)
            self.assertIsInstance(rebuilt.layers[0].gene_q_norm, torch.nn.Identity)
            torch.save(dict(schema_version=5, config=config), path)
            with self.assertRaisesRegex(ValueError, "attention-normalization"):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()

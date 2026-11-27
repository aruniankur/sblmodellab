import tempfile
import unittest
from pathlib import Path

import torch

from models.model import ExternalPerturbAttention
from models.training.checkpoints import load_checkpoint
from models.training.config import TrainingConfig, parse_args
from models.training.modeling import create_model


class ChemicalFFNResidualTests(unittest.TestCase):
    def test_identity_path_survives_zero_ffn_and_carries_gradients(self):
        torch.manual_seed(4)
        block = ExternalPerturbAttention(16, 2, dropout=0, normalization=False)
        with torch.no_grad():
            block.chem_ffn[2].weight.zero_()
            block.chem_ffn[2].bias.zero_()
        before, after = [], []
        h1 = block.chem_ffn.register_forward_pre_hook(lambda m, args: before.append(args[0]))
        h2 = block.chem_ffn_norm.register_forward_pre_hook(lambda m, args: after.append(args[0]))
        x, chem = torch.randn(1, 5, 16), torch.randn(1, 3, 16)
        block(x, chem)
        torch.testing.assert_close(after[-1], before[-1])
        after[-1].square().mean().backward()
        self.assertTrue(torch.isfinite(block.kv_linear.weight.grad).all())
        self.assertGreater(float(block.kv_linear.weight.grad.norm()), 0)
        block.use_chem_ffn_residual = False
        block(x, chem)
        torch.testing.assert_close(after[-1], torch.zeros_like(after[-1]))
        h1.remove()
        h2.remove()

    def test_schema_five_preserves_old_forward_and_schema_six_requires_flag(self):
        cfg = TrainingConfig(dim=16, conc_dim=4, heads=2, layers=1, dropout=0,
                             chem_ffn_residual=False)
        original = create_model(cfg, 7, 5).eval()
        config = cfg.to_dict()
        config.pop("chem_ffn_residual")
        args = (torch.rand(1, 7, 8), torch.rand(1, 3, 5), torch.rand(1, 2, 2))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            torch.save(dict(schema_version=5, config=config, model=original.state_dict()), path)
            _, state = load_checkpoint(path)
            self.assertFalse(state["config"]["chem_ffn_residual"])
            rebuilt = create_model(TrainingConfig(**state["config"]), 7, 5).eval()
            rebuilt.load_state_dict(state["model"], strict=True)
            torch.testing.assert_close(original(*args), rebuilt(*args), rtol=0, atol=0)
            torch.save(dict(schema_version=6, config=config), path)
            with self.assertRaisesRegex(ValueError, "chemical-FFN-residual"):
                load_checkpoint(path)

    def test_new_run_default_and_ablation(self):
        _, cfg = parse_args([])
        self.assertTrue(cfg.chem_ffn_residual)
        self.assertTrue(create_model(cfg, 7, 5).layers[0].use_chem_ffn_residual)
        _, cfg = parse_args(["--no-chem-ffn-residual"])
        self.assertFalse(cfg.chem_ffn_residual)
        cfg.chem_ffn_residual = "yes"
        with self.assertRaisesRegex(ValueError, "chem_ffn_residual"):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()

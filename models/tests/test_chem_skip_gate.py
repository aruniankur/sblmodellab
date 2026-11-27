import tempfile
import unittest
from pathlib import Path

import torch

from models.model import ExternalPerturbAttention
from models.training.checkpoints import load_checkpoint
from models.training.config import TrainingConfig, parse_args
from models.training.metrics import parameter_counts
from models.training.modeling import create_model


class ChemicalSkipGateTests(unittest.TestCase):
    def test_gate_preserves_direct_message_with_zero_gene_ffn_and_learns(self):
        torch.manual_seed(13)
        block = ExternalPerturbAttention(16, 2, dropout=0, normalization=False)
        with torch.no_grad():
            block.gene_ffn[2].weight.zero_()
            block.gene_ffn[2].bias.zero_()
        context = []
        handle = block.chem_ffn_norm.register_forward_hook(lambda m, args, y: context.append(y))
        x, chem = torch.randn(1, 7, 16), torch.randn(1, 4, 16)
        y = block(x, chem)
        self.assertAlmostEqual(float(block.chem_skip_gate_logit.sigmoid()), .1, places=7)
        torch.testing.assert_close(y, x + .1 * context[0])
        (y - x).square().mean().backward()
        self.assertGreater(float(block.chem_skip_gate_logit.grad.abs()), 0)
        self.assertGreater(float(block.kv_linear.weight.grad.norm()), 0)
        before = block.chem_skip_gate_logit.detach().clone()
        torch.optim.SGD(block.parameters(), lr=.1).step()
        self.assertFalse(torch.equal(before, block.chem_skip_gate_logit.detach()))
        self.assertTrue(0 < float(block.chem_skip_gate_logit.sigmoid()) < 1)
        handle.remove()

    def test_three_independent_gates_get_gradients_and_update(self):
        torch.manual_seed(21)
        cfg = TrainingConfig(heads=2, head_dim=8, layers=3, dropout=0.)
        model = create_model(cfg, 24, 6)
        gates = [block.chem_skip_gate_logit for block in model.layers]
        self.assertEqual(len({g.data_ptr() for g in gates}), 3)
        groups = parameter_counts(model)
        for i in range(3):
            self.assertEqual(groups[f'layer{i}.chem_skip_gate_logit'], 1)
        before = [g.detach().clone() for g in gates]
        y = model(torch.rand(1,24,32), torch.rand(1,5,6), torch.rand(1,12,2))
        (y - torch.rand_like(y)).square().mean().backward()
        for g in gates:
            self.assertTrue(torch.isfinite(g.grad))
            self.assertGreater(float(g.grad.abs()), 0)
        torch.optim.AdamW(model.parameters(), lr=.001).step()
        self.assertTrue(all(not torch.equal(old, g.detach()) for old,g in zip(before,gates)))

    def test_old_schema_eight_preserves_keys_forward_and_optimizer(self):
        cfg = TrainingConfig(heads=2, head_dim=8, layers=2, dropout=0., chem_skip_gate=False)
        model = create_model(cfg, 12, 6).eval()
        optimizer = torch.optim.AdamW(model.parameters())
        args = (torch.rand(1,12,16), torch.rand(1,4,6), torch.rand(1,8,2))
        model(*args).square().mean().backward();optimizer.step()
        config = cfg.to_dict();config.pop('chem_skip_gate');config.pop('chem_skip_init')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'last.pt'
            torch.save(dict(schema_version=8, config=config, model=model.state_dict(), optimizer=optimizer.state_dict()), path)
            _, state = load_checkpoint(path)
            self.assertFalse(state['config']['chem_skip_gate'])
            restored = create_model(TrainingConfig(**state['config']),12,6).eval()
            restored.load_state_dict(state['model'],strict=True)
            resumed_optimizer = torch.optim.AdamW(restored.parameters())
            resumed_optimizer.load_state_dict(state['optimizer'])
            torch.testing.assert_close(model(*args),restored(*args),rtol=0,atol=0)
            self.assertFalse(any('chem_skip_gate_logit' in key for key in restored.state_dict()))
            restored(*args).square().mean().backward();resumed_optimizer.step()

    def test_new_schema_restores_learned_gate_and_requires_config(self):
        cfg = TrainingConfig(heads=2, head_dim=8, layers=2, dropout=0.)
        model = create_model(cfg,12,6)
        with torch.no_grad():model.layers[0].chem_skip_gate_logit.fill_(.7)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'last.pt'
            state=dict(schema_version=9,config=cfg.to_dict(),model=model.state_dict())
            torch.save(state,path)
            _,saved=load_checkpoint(path)
            restored=create_model(TrainingConfig(**saved['config']),12,6)
            restored.load_state_dict(saved['model'],strict=True)
            torch.testing.assert_close(restored.layers[0].chem_skip_gate_logit,model.layers[0].chem_skip_gate_logit)
            state['config'].pop('chem_skip_gate');torch.save(state,path)
            with self.assertRaisesRegex(ValueError,'chemical-skip-gate'):load_checkpoint(path)

    def test_defaults_ablation_and_invalid_initialization(self):
        _,cfg=parse_args([])
        self.assertTrue(cfg.chem_skip_gate);self.assertEqual(cfg.chem_skip_init,.1)
        _,cfg=parse_args(['--no-chem-skip-gate','--chem-skip-init','.25'])
        self.assertFalse(cfg.chem_skip_gate)
        self.assertIsNone(create_model(cfg,7,5).layers[0].chem_skip_gate_logit)
        for value in (0,1,-.1,float('nan'),float('inf'),True,'.1'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError,'chem_skip_init'):
                    TrainingConfig(chem_skip_init=value).validate()
                with self.assertRaisesRegex(ValueError,'chem_skip_init'):
                    ExternalPerturbAttention(16,2,chem_skip_init=value)


if __name__ == '__main__':unittest.main()

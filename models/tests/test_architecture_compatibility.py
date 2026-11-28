import tempfile, unittest
from pathlib import Path
import torch
from models.training.config import TrainingConfig
from models.training.modeling import create_model
from models.training.checkpoints import load_checkpoint

class ArchitectureCompatibilityTests(unittest.TestCase):
    def test_defaults_and_legacy_architecture_inference(self):
        self.assertEqual(TrainingConfig().ffn_multiplier, 2)
        self.assertFalse(TrainingConfig().joint_normalization)
        for multiplier, joint in ((2, False), (4, True)):
            with self.subTest(multiplier=multiplier), tempfile.TemporaryDirectory() as directory:
                cfg = TrainingConfig(model_type='full', heads=2, head_dim=8, layers=2,
                                     dropout=0., ffn_multiplier=multiplier, joint_normalization=joint,
                                     legacy_attention_norms=True)
                model = create_model(cfg, 12, 6).eval()
                self.assertEqual(model.layers[0].chem_ffn[0].out_features, cfg.dim * multiplier)
                self.assertEqual(isinstance(model.layers[0].joint_norm, torch.nn.LayerNorm), joint)
                saved = cfg.to_dict()
                saved.pop('ffn_multiplier')
                saved.pop('joint_normalization')
                path = Path(directory) / 'last.pt'
                torch.save(dict(schema_version=7, config=saved, model=model.state_dict()), path)
                _, state = load_checkpoint(path)
                self.assertEqual(state['config']['ffn_multiplier'], multiplier)
                self.assertEqual(state['config']['joint_normalization'], joint)
                restored = create_model(TrainingConfig(**state['config']), 12, 6).eval()
                restored.load_state_dict(state['model'], strict=True)
                args = (torch.rand(1,12,16), torch.rand(1,4,6), torch.rand(1,8,2))
                with torch.no_grad():
                    torch.testing.assert_close(model(*args), restored(*args), rtol=0, atol=0)

    def test_new_schema_requires_architecture_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            config = TrainingConfig().to_dict()
            config.pop('ffn_multiplier')
            path = Path(directory) / 'last.pt'
            torch.save(dict(schema_version=8, config=config), path)
            with self.assertRaisesRegex(ValueError, 'FFN-width'):
                load_checkpoint(path)

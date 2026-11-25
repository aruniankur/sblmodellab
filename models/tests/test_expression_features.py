"""Tests for log-scale, cell-order-invariant magnitude features and model fusion."""
import math
import unittest

import torch

from models.expression_features import GENE_STAT_NAMES, gene_expression_statistics
from models.control_model import ControlPerturbationModel
from models.model import PerturbationModel


class ExpressionFeatureTests(unittest.TestCase):
    def test_five_statistics_are_on_the_supplied_log_scale(self):
        x = torch.tensor([[[0., 1., 2., 3.], [0., 0., 0., 0.],
                           [0., 0., 2., 0.], [2., 2., 2., 2.]]])
        expected = torch.tensor([[[1.5, math.sqrt(1.25), .75, 2., 2.8],
                                  [0., 0., 0., 0., 0.],
                                  [.5, math.sqrt(.75), .25, 2., 2.],
                                  [2., 0., 1., 2., 2.]]])
        actual = gene_expression_statistics(x)
        self.assertEqual(len(GENE_STAT_NAMES), 5)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, gene_expression_statistics(x[..., [3, 1, 0, 2]]))

    def test_one_cell_empty_input_and_finite_input_gradients(self):
        x = torch.tensor([[[0.], [2.]]], requires_grad=True)
        actual = gene_expression_statistics(x)
        torch.testing.assert_close(actual, torch.tensor([[[0., 0., 0., 0., 0.], [2., 0., 1., 2., 2.]]]))
        actual.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for invalid in (torch.zeros(2, 4), torch.zeros(1, 4, 0)):
            with self.assertRaises(ValueError):
                gene_expression_statistics(invalid)

    def test_fp32_statistics_inside_autocast(self):
        x = torch.tensor([[[0., .1, .3, .9]]])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = gene_expression_statistics(x)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, gene_expression_statistics(x))

    def test_merger_receives_five_features_and_models_keep_cell_order_invariance(self):
        torch.manual_seed(7)
        control = torch.rand(1, 16, 8)
        control[control < .2] = 0
        coords = torch.randn(1, 4, 2)
        chem = torch.rand(1, 3, 5)
        for cls in (ControlPerturbationModel, PerturbationModel):
            kwargs = dict(dim=16, conc_dim=4, num_heads=2, num_layers=1,
                          dropout=0., num_of_genes=16)
            if cls is PerturbationModel:
                kwargs["chem_dim"] = 5
            model = cls(**kwargs).eval()
            self.assertEqual(model.merger.in_features, 16 + 4 + 5)
            captured = []
            handle = model.merger.register_forward_pre_hook(lambda m, args: captured.append(args[0].detach().clone()))
            def predict(context):
                args = (context, coords) if cls is ControlPerturbationModel else (context, chem, coords)
                return model(*args, query_chunk=5, readout_chunk=2)
            first = predict(control)
            torch.testing.assert_close(captured[0][..., -5:], gene_expression_statistics(control))
            permuted = predict(control[..., [4, 1, 7, 3, 0, 6, 5, 2]])
            torch.testing.assert_close(first, permuted, atol=1e-6, rtol=1e-5)
            scaled = predict(control * 3)
            self.assertGreater(float((first - scaled).abs().max()), 1e-5)
            first.square().mean().backward()
            self.assertTrue(torch.isfinite(model.merger.weight.grad).all())
            self.assertGreater(float(model.merger.weight.grad[:, -5:].abs().sum()), 0)
            handle.remove()
            old = cls(**kwargs, gene_statistics=False).eval()
            self.assertEqual(old.merger.in_features, 16 + 4)
            self.assertEqual(sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in old.parameters()), 5 * 16)
            with torch.no_grad():
                args1 = (control, coords) if cls is ControlPerturbationModel else (control, chem, coords)
                args2 = (control * 3, coords) if cls is ControlPerturbationModel else (control * 3, chem, coords)
                torch.testing.assert_close(old(*args1), old(*args2), atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

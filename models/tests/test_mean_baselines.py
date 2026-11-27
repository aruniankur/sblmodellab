import csv
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from models.training.metrics import PassLog
from models.training.modeling import mean_baseline_losses


class MeanBaselineTests(unittest.TestCase):
    def example(self):
        return SimpleNamespace(
            target=torch.tensor([[1., 0.], [3., 2.], [5., 4.]], requires_grad=True),
            control=torch.tensor([[0., 2., 4., 6.], [2., 4., 6., 8.]], requires_grad=True))

    def test_means_are_per_gene_and_use_model_loss_scale(self):
        ex = self.example()
        rng = torch.get_rng_state().clone()
        for exponent in (0., .5, 1.):
            target, control = mean_baseline_losses(ex, SimpleNamespace(scale_exponent=exponent))
            # Target mean = [3, 2]; input mean = [3, 5].
            self.assertAlmostEqual(target, (16 / 3) / 2**exponent, places=5)
            self.assertAlmostEqual(control, (43 / 3) / 2**exponent, places=5)
        self.assertIsNone(ex.target.grad)
        self.assertIsNone(ex.control.grad)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_baselines_stay_fp32_inside_autocast(self):
        ex = self.example()
        cfg = SimpleNamespace(scale_exponent=.5)
        reference = mean_baseline_losses(ex, cfg)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = mean_baseline_losses(ex, cfg)
        self.assertEqual(reference, actual)

    def test_old_csv_upgrades_with_blanks_and_trims_uncommitted_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'log.csv'
            log = PassLog(path, ['gene_embedding'])
            fields = [f for f in log.fields if f not in ('t_mean_loss', 'i_mean_loss')]
            log.close()
            row = {f: '' for f in fields}
            row.update(backward_pass='1', loss='4.123456789', grad_gene_embedding='0.00000001')
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
                writer.writerow({**row, 'backward_pass': '999'})
            log = PassLog(path, ['gene_embedding'], committed_passes=1)
            log.write({**row, 'backward_pass': 2, 't_mean_loss': 1.25,
                       'i_mean_loss': 3.5, 'grad_total': 1e-8, 'grad_gene_embedding': 1e-8})
            log.close()
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r['backward_pass'] for r in rows], ['1', '2'])
            self.assertEqual(rows[0]['loss'], '4.123456789')
            self.assertEqual(rows[0]['t_mean_loss'], '')
            self.assertEqual(rows[0]['i_mean_loss'], '')
            self.assertEqual(rows[1]['t_mean_loss'], '1.25')
            self.assertEqual(rows[1]['i_mean_loss'], '3.5')
            self.assertEqual(rows[1]['grad_gene_embedding'], '0.00000001')

    def test_csv_migration_still_rejects_different_parameter_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'log.csv'
            log = PassLog(path, ['gene_embedding'])
            log.close()
            with self.assertRaisesRegex(ValueError, 'parameter groups'):
                PassLog(path, ['different_model'])


if __name__ == '__main__':
    unittest.main()

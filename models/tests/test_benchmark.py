"""Capacity-sweep orchestration and real tiny training-worker tests."""
import csv
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import test_training as fixtures
from models.benchmark import inclusive_range, isolated_trial, sweep
from models.training.capacity import run_trial
from models.training.metrics import PassLog


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PipelineTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def config(self, **kwargs):
        return self.fixture.cfg(mode="single", file=str(self.fixture.data / "plate1_A.h5ad"), **kwargs)

    def test_ranges_and_invalid_heads(self):
        self.assertEqual(inclusive_range(16, 32, 8), [16, 24, 32])
        self.assertEqual(inclusive_range(16, 30, 8), [16, 24])
        for values in ((0, 32, 8), (32, 16, 8), (16, 32, 0)):
            with self.assertRaises(ValueError):
                inclusive_range(*values)
        row = isolated_trial(self.config(model_type="full", dim=17), {}, 1)
        self.assertEqual(row["status"], "invalid")

    def test_real_isolated_control_training(self):
        result = isolated_trial(self.config(), {"steps": 1, "warmup": 1,
                                "max_vram_fraction": .99}, 60)
        self.assertIn(result["status"], ("fits", "tight", "cpu_completed"), result)
        self.assertEqual(result["completed_steps"], 1)
        self.assertEqual(result["n_genes"], 16)
        self.assertEqual(result["micro_batches_per_unit"], 2)
        self.assertGreater(result["seconds_per_unit"], 0)
        self.assertGreater(result["parameters"], 0)

    def test_full_worker_performs_training(self):
        with patch("models.training.capacity.load_chemistry", return_value=fixtures.FakeChemistry()):
            result = run_trial(self.config(model_type="full"), steps=1, warmup=0)
        self.assertIn(result["status"], ("fits", "tight", "cpu_completed"), result)
        self.assertEqual(result["completed_steps"], 1)
        self.assertGreater(result["grad_total"], 0)

    def test_timeout_kills_worker(self):
        process = Mock()
        process.communicate.side_effect = [subprocess.TimeoutExpired("worker", .1), ("", "")]
        with patch("models.benchmark.subprocess.Popen", return_value=process):
            result = isolated_trial(self.config(), {}, .1)
        self.assertEqual(result["status"], "timeout")
        process.kill.assert_called_once()
        self.assertEqual(process.communicate.call_count, 2)

    def test_interrupt_kills_worker(self):
        process = Mock()
        process.communicate.side_effect = [KeyboardInterrupt, ("", "")]
        with patch("models.benchmark.subprocess.Popen", return_value=process):
            with self.assertRaises(KeyboardInterrupt):
                isolated_trial(self.config(), {}, 1)
        process.kill.assert_called_once()

    def test_sweep_continues_after_oom_and_reports_frontier(self):
        output = self.fixture.root / "report"
        results = [dict(dim=32, head_dim=16, layers=1, status="oom"), dict(dim=32, head_dim=16, layers=2, status="fits"),
                   dict(dim=64, head_dim=32, layers=1, status="fits"), dict(dim=64, head_dim=32, layers=2, status="tight")]
        with patch("models.benchmark.isolated_trial", side_effect=results) as worker:
            rows = sweep(self.config(model_type="full"), [16, 32], [1, 2], output=output)
        self.assertEqual(worker.call_count, 4)
        for call in worker.call_args_list:
            candidate = call.args[0]
            self.assertEqual(candidate.dim, candidate.heads * candidate.head_dim)
            self.assertEqual(candidate.conc_dim, candidate.head_dim)
            self.assertEqual(candidate.grad_clip, 3.)
        self.assertEqual(len(rows), 4)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["largest_fitting_dim_by_layers"], {"1": 64, "2": 32})
        with (output / "results.csv").open(newline="") as fh:
            self.assertEqual(len(list(csv.DictReader(fh))), 4)
        with self.assertRaises(FileExistsError):
            sweep(self.config(), [16], [1], output=output)

    def test_control_layers_are_not_repeated(self):
        with patch("models.benchmark.isolated_trial", return_value=dict(dim=32, head_dim=16, layers=1, status="fits")) as worker:
            sweep(self.config(), [16], [1, 2, 3], output=self.fixture.root / "control_report")
        worker.assert_called_once()

    def test_benchmark_cli_uses_per_head_ranges(self):
        from models.benchmark import parse_args
        _, args = parse_args(["--heads", "8", "--head-dim-start", "32",
                              "--head-dim-end", "128", "--head-dim-step", "32",
                              "--layers-start", "1", "--layers-end", "3"])
        self.assertEqual((args.heads, args.head_dim_start, args.head_dim_end, args.head_dim_step),
                         (8, 32, 128, 32))
        self.assertFalse(hasattr(args, "conc_dim"))
        self.assertFalse(hasattr(args, "dim_start"))

    def test_gradient_csv_uses_eight_decimal_places_only_for_gradients(self):
        path = self.fixture.root / "log.csv"
        log = PassLog(path, ["gene_embedding"])
        values = {key: "" for key in log.fields}
        values.update(grad_total=1.23456789123, grad_gene_embedding=1.23456789e-8, loss=1.23456789123)
        log.write(values)
        log.close()
        with path.open(newline="") as fh:
            row = next(csv.DictReader(fh))
        self.assertEqual(row["grad_total"], "1.23456789")
        self.assertEqual(row["grad_gene_embedding"], "0.00000001")
        self.assertEqual(row["loss"], "1.23456789123")


if __name__ == "__main__":
    unittest.main()

"""Small synthetic-data checks; never open or modify the real Tahoe dataset."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from models.tahoe_data import BackedFile
from models.training.checkpoints import load_checkpoint
from models.training.config import TrainingConfig, parse_args
from models.training.data import epoch_units, make_plan, sample_example, targets
from models.training.engine import train
from models.training.metrics import gradient_norms, parameter_counts
from models.training.modeling import create_model


class FakeChemistry:
    dim = 4

    def collate(self, drugs, doses, device=None, mode="all"):
        tokens = torch.ones(len(drugs), 3, 5, device=device)
        for i, dose in enumerate(doses):
            tokens[i, :, -1] = dose
        return tokens, torch.ones(len(drugs), 1, 1, 3, dtype=torch.bool, device=device)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(prefix="tahoe_pipeline_")
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        for plate in (1, 2):
            for line in ("A", "B"):
                self.write_file(plate, line)

    def tearDown(self):
        self.temp.cleanup()

    def write_file(self, plate, line, nan=False):
        rng = np.random.default_rng(plate)
        counts = rng.poisson(1, size=(24, 16)).astype(np.float32)
        counts[0] = 0
        obs = pd.DataFrame({
            "sample": np.repeat(["dmso1", "dmso2", "drug1", "drug2"], 6),
            "drug": np.repeat(["DMSO_TF", "DMSO_TF", "drugA", "drugB"], 6),
            "canonical_smiles": ["C"] * 24,
            "concentration": np.repeat([0., 0., .5, 5.], 6),
            "coord_1": np.full(24, np.nan) if nan else rng.normal(size=24),
            "coord_2": np.full(24, np.nan) if nan else rng.normal(size=24),
        }, index=[f"cell{i}" for i in range(24)])
        var = pd.DataFrame(index=[f"gene{i}" for i in range(16)])
        path = self.data / f"plate{plate}_{line}.h5ad"
        ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var).write_h5ad(path)
        return path

    def cfg(self, **values):
        defaults = dict(sampling_strategy="legacy", model_type="control", root=str(self.data), dim=16, conc_dim=4,
                        heads=2, layers=1, epochs=1, batch_size=1, accumulate_batches=True, n_control=8, n_target=8, min_cells=2,
                        passes_per_cell_line=1, batches_per_unit=2, file_cache=2,
                        query_chunk=5, readout_chunk=3, amp="off", dropout=0., grad_clip=3., save_every=1)
        return TrainingConfig(**{**defaults, **values})

    def test_duplicate_rows_and_same_well_overlap(self):
        path = self.data / "plate1_A.h5ad"
        backed = BackedFile(str(path))
        try:
            expected = ad.read_h5ad(path).X.toarray()[[4, 1, 4, 0, 1]]
            np.testing.assert_array_equal(backed.read_rows(np.array([4, 1, 4, 0, 1])), expected)
            example = sample_example(backed, "dmso1", self.cfg(), np.random.default_rng(0))
            self.assertEqual(example.control.shape, (16, 8))
            self.assertEqual(example.target.shape, (8, 16))
            self.assertTrue(torch.isfinite(example.control).all())
            self.assertEqual(example.drug, "DMSO_TF")
        finally:
            backed.close()

    def test_control_plan_uses_all_eligible_lines_and_only_dmso(self):
        self.write_file(1, "NO_COORDS", nan=True)
        self.write_file(13, "HELD_OUT")
        cfg = self.cfg()
        plan = make_plan(cfg)
        self.assertEqual(plan.train_lines, ["A", "B"])
        self.assertFalse(plan.val_lines)
        units = epoch_units(plan, cfg, np.random.default_rng(2))
        self.assertEqual(len(units), 2)
        self.assertTrue(all(u["plate"] <= 12 for u in units))
        self.assertTrue(all(set(u["samples"]) == {"dmso1", "dmso2"} for u in units))

    def test_full_split_and_single_file_schedule(self):
        cfg = self.cfg(model_type="full", val_cell_lines=1, passes_per_cell_line=2)
        plan = make_plan(cfg)
        self.assertEqual(len(plan.val_lines), 1)
        self.assertTrue(set(plan.val_lines).isdisjoint(plan.train_lines))
        units = epoch_units(plan, cfg, np.random.default_rng(1))
        self.assertEqual({u["plate"] for u in units}, {1, 2})
        self.assertTrue(all(set(u["samples"]) == {"drug1", "drug2"} for u in units))
        cfg.mode, cfg.file = "single", str(self.data / "plate1_A.h5ad")
        plan = make_plan(cfg)
        self.assertFalse(plan.val_lines)
        units = epoch_units(plan, cfg, np.random.default_rng(1))
        self.assertEqual(len(units), 2)
        self.assertTrue(all(u["path"] == cfg.file for u in units))

    def test_gradient_metrics_are_per_backward_not_accumulated(self):
        model = torch.nn.Linear(2, 1, bias=False)
        groups = parameter_counts(model)
        x = torch.tensor([[2., 3.]])
        (model(x).sum() / 2).backward()
        _, first = gradient_norms(model, groups, multiplier=2)
        before = {n: p.grad.clone() for n, p in model.named_parameters()}
        (model(x * 2).sum() / 2).backward()
        _, second = gradient_norms(model, groups, before, multiplier=2)
        self.assertAlmostEqual(second, 2 * first, places=5)

    def test_control_training_csv_and_resume_match_uninterrupted(self):
        with patch("models.training.checkpoints.TRAINED_ROOT", self.root / "runs"), patch(
                "models.molformer_atoms.AtomEmbeddingStore") as no_cache:
            run = train(self.cfg())
            _, reference = load_checkpoint(run)
            interrupted = train(self.cfg(max_steps=1))
            # Simulate a CSV row written after the last checkpoint but before a crash.
            log_path = interrupted / "log.csv"
            with log_path.open(newline="") as fh:
                rows = list(csv.DictReader(fh))
                fields = list(rows[0])
            extra = {**rows[-1], "backward_pass": "999"}
            with log_path.open("a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=fields).writerow(extra)
            train(resume=str(interrupted), additional_epochs=1)
            _, resumed = load_checkpoint(interrupted)
            for name, value in reference["model"].items():
                torch.testing.assert_close(value, resumed["model"][name], rtol=0, atol=0)
            with log_path.open(newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([int(r["backward_pass"]) for r in rows], [1, 2, 3, 4])
            self.assertTrue(all(r["drug"] == "DMSO_TF" for r in rows))
            self.assertIn("grad_gene_embedding", rows[0])
            for row in rows:
                self.assertTrue(np.isfinite(float(row["t_mean_loss"])))
                self.assertTrue(np.isfinite(float(row["i_mean_loss"])))
                self.assertLessEqual(float(row["t_mean_loss"]), float(row["i_mean_loss"]) + 1e-5)
            self.assertTrue(all(np.isfinite(float(r["grad_total"])) for r in rows))
            no_cache.assert_not_called()
            self.assertTrue(run.name.endswith("_control"))
            self.assertTrue((run / "info.json").exists())
            self.assertFalse((run / "grad.csv").exists())

    def test_resume_old_csv_adds_mean_losses_and_tqdm_shows_them(self):
        import io
        from contextlib import redirect_stderr
        stream = io.StringIO()
        with patch("models.training.checkpoints.TRAINED_ROOT", self.root / "runs"), redirect_stderr(stream):
            run = train(self.cfg(max_steps=1))
            path = run / "log.csv"
            with path.open(newline="") as fh:
                reader = csv.DictReader(fh)
                fields = [f for f in reader.fieldnames if f not in ("t_mean_loss", "i_mean_loss")]
                old_rows = [{k: v for k, v in row.items() if k in fields} for row in reader]
            with path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(old_rows)
                writer.writerow({**old_rows[-1], "backward_pass": "999"})
            state = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            state["schema_version"] = 6
            torch.save(state, run / "last.pt")
            train(resume=str(run), additional_epochs=1)
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual([int(r["backward_pass"]) for r in rows], [1, 2, 3, 4])
        self.assertTrue(all(r["t_mean_loss"] == r["i_mean_loss"] == "" for r in rows[:2]))
        self.assertTrue(all(np.isfinite(float(r[k])) for r in rows[2:]
                            for k in ("t_mean_loss", "i_mean_loss")))
        self.assertIn("t_mean_loss=", stream.getvalue())
        self.assertIn("i_mean_loss=", stream.getvalue())
        _, state = load_checkpoint(run)
        self.assertEqual(state["schema_version"], 12)

    def test_full_single_training_and_additional_epoch_resume(self):
        cfg = self.cfg(model_type="full", mode="single", file=str(self.data / "plate1_A.h5ad"))
        with patch("models.training.checkpoints.TRAINED_ROOT", self.root / "runs"), patch(
                "models.training.engine.load_chemistry", return_value=FakeChemistry()):
            run = train(cfg)
            train(resume=str(run), additional_epochs=1)
        _, state = load_checkpoint(run)
        self.assertEqual(state["progress"]["epoch"], 2)
        self.assertEqual(state["progress"]["backward_pass"], 4)
        self.assertEqual(state["progress"]["optimizer_step"], 4)
        with (run / "log.csv").open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual({r["file"] for r in rows}, {"plate1_A.h5ad"})
        self.assertEqual({r["drug"] for r in rows}, {"drugA", "drugB"})
        info = json.loads((run / "info.json").read_text())
        self.assertEqual(info["status"], "finished")
        self.assertIn("reconstruction_probe", info["epochs"][0])
        self.assertEqual(len(info["epochs"][0]["chem_skip_gates"]), 1)
        self.assertIn("grad_layer0.chem_skip_gate_logit", rows[0])
        self.assertTrue((run / "best.pt").exists())
        self.assertTrue(run.name.endswith("_full"))

    def test_model_chunk_parity_for_both_architectures(self):
        for model_type in ("control", "full"):
            cfg = self.cfg(model_type=model_type)
            model = create_model(cfg, 16, 5).eval()
            context = torch.rand(1, 16, 8)
            coords = torch.rand(1, 4, 2)
            inputs = (context, coords) if model_type == "control" else (
                context, torch.rand(1, 3, 5), coords, torch.ones(1, 1, 1, 3, dtype=torch.bool))
            reference = model(*inputs)
            chunked = model(*inputs, query_chunk=5, readout_chunk=2)
            torch.testing.assert_close(reference, chunked, atol=1e-6, rtol=1e-5)
            model.zero_grad()
            reference.square().mean().backward()
            grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            model.zero_grad()
            model(*inputs, query_chunk=5, readout_chunk=2).square().mean().backward()
            for name, p in model.named_parameters():
                if name in grads:
                    torch.testing.assert_close(grads[name], p.grad, atol=1e-6, rtol=1e-4)

    def test_normalization_locations_and_gradients(self):
        for model_type in ("control", "full"):
            cfg = self.cfg(model_type=model_type, layers=2)
            model = create_model(cfg, 16, 5)
            norms = {name: module for name, module in model.named_modules()
                     if isinstance(module, torch.nn.LayerNorm)}
            expected = {"merger_norm"}
            if model_type == "full":
                expected |= {"chem_norm"}
                self.assertFalse(hasattr(model, "final_norm"))
                expected |= {f"layers.{i}.{name}" for i in range(2)
                             for name in ("chem_ffn_norm", "gene_ffn_norm")}
            self.assertEqual(set(norms), expected)
            for name, module in norms.items():
                self.assertEqual(module.normalized_shape, (32,) if name.endswith("joint_norm") else (16,))
            chemistry_calls = []
            handle = model.chem_norm.register_forward_hook(lambda *args: chemistry_calls.append(1)) if model_type == "full" else None
            context = torch.rand(1, 16, 8)
            coords = torch.rand(1, 4, 2)
            args = (context, coords) if model_type == "control" else (
                context, torch.rand(1, 3, 5), coords,
                torch.ones(1, 1, 1, 3, dtype=torch.bool))
            output = model(*args, query_chunk=5, readout_chunk=2)
            self.assertEqual(output.shape, (1, 4, 16))
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            for module in norms.values():
                for parameter in module.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            if handle is not None:
                self.assertEqual(len(chemistry_calls), 1)
                handle.remove()
            legacy = create_model(self.cfg(model_type=model_type, normalization="none"), 16, 5)
            self.assertFalse(any(isinstance(m, torch.nn.LayerNorm) for m in legacy.modules()))
            self.assertFalse(any("norm." in key for key in legacy.state_dict()))

    def test_normalized_models_have_finite_bf16_gradients(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            self.skipTest("GPU does not support BF16")
        for model_type in ("control", "full"):
            model = create_model(self.cfg(model_type=model_type, layers=2), 16, 5).to(device)
            context = torch.rand(1, 16, 8, device=device)
            coords = torch.rand(1, 4, 2, device=device)
            args = (context, coords) if model_type == "control" else (
                context, torch.rand(1, 3, 5, device=device), coords,
                torch.tensor([True, True, False], device=device)[None, None, None, :])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                output = model(*args, query_chunk=5, readout_chunk=2)
            loss = output.float().square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_pre_normalization_checkpoint_migrates_without_architecture_change(self):
        with patch("models.training.checkpoints.TRAINED_ROOT", self.root / "runs"):
            run = train(self.cfg(normalization="none", gene_statistics=False))
            original = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            original["schema_version"] = 1
            original["config"].pop("normalization")
            original["config"].pop("gene_statistics")
            torch.save(original, run / "last.pt")
            _, restored = load_checkpoint(run)
            self.assertEqual(restored["config"]["normalization"], "none")
            model = create_model(TrainingConfig(**restored["config"]), 16)
            model.load_state_dict(restored["model"], strict=True)
            train(resume=str(run), additional_epochs=1)
        _, resumed = load_checkpoint(run)
        self.assertEqual(resumed["schema_version"], 12)
        self.assertEqual(resumed["config"]["normalization"], "none")
        self.assertEqual(resumed["progress"]["epoch"], 2)
        self.assertFalse(any("norm." in key for key in resumed["model"]))

    def test_pre_statistics_normalized_checkpoint_preserves_original_merger(self):
        with patch("models.training.checkpoints.TRAINED_ROOT", self.root / "runs"):
            run = train(self.cfg(gene_statistics=False))
            original = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            original["schema_version"] = 2
            original["config"].pop("gene_statistics")
            torch.save(original, run / "last.pt")
            _, restored = load_checkpoint(run)
            self.assertFalse(restored["config"]["gene_statistics"])
            self.assertEqual(restored["config"]["normalization"], "layernorm")
            self.assertEqual(restored["model"]["merger.weight"].shape[1], 20)
            train(resume=str(run), additional_epochs=1)
        _, resumed = load_checkpoint(run)
        self.assertEqual(resumed["schema_version"], 12)
        self.assertFalse(resumed["config"]["gene_statistics"])
        self.assertEqual(resumed["model"]["merger.weight"].shape[1], 20)
        self.assertEqual(resumed["progress"]["epoch"], 2)

    def test_remaining_training_combinations(self):
        configs = [self.cfg(model_type="full", val_cell_lines=1),
                   self.cfg(mode="single", file=str(self.data / "plate1_A.h5ad"))]
        with patch("models.training.checkpoints.TRAINED_ROOT", self.root / "runs"), patch(
                "models.training.engine.load_chemistry", side_effect=lambda cfg:
                FakeChemistry() if cfg.model_type == "full" else None):
            for cfg in configs:
                run = train(cfg)
                _, state = load_checkpoint(run)
                self.assertEqual(state["progress"]["epoch"], 1)
                with (run / "log.csv").open(newline="") as fh:
                    rows = list(csv.DictReader(fh))
                expected_drugs = {"drugA", "drugB"} if cfg.model_type == "full" else {"DMSO_TF"}
                self.assertEqual({r["drug"] for r in rows}, expected_drugs)
                if cfg.model_type == "full":
                    self.assertTrue(set(r["cell_line"] for r in rows).isdisjoint(state["plan"]["val_lines"]))
                    self.assertIn("validation", state["info"]["epochs"][0])
                else:
                    self.assertEqual({r["file"] for r in rows}, {"plate1_A.h5ad"})

    def test_setup_failure_is_recorded_in_run_metadata(self):
        with patch("models.training.checkpoints.TRAINED_ROOT", self.root / "runs"), patch(
                "models.training.engine.load_chemistry", side_effect=FileNotFoundError("missing chemistry cache")):
            with self.assertRaises(FileNotFoundError):
                train(self.cfg(model_type="full"))
        runs = list((self.root / "runs").iterdir())
        info = json.loads((runs[0] / "info.json").read_text())
        self.assertEqual(info["status"], "failed")
        self.assertIn("missing chemistry cache", info["error"])

    def test_interactive_new_and_resume_menus(self):
        from models.run import main
        with patch("models.run.check_environment", return_value="off"), patch(
                "builtins.input", side_effect=["1", "2", "1", "8", "128", "1", "4", "3"]), patch(
                "models.training.engine.train") as training:
            main()
            cfg = training.call_args.args[0]
            self.assertEqual(cfg.batch_size, 4)
            self.assertFalse(cfg.accumulate_batches)
            self.assertEqual((cfg.model_type, cfg.mode, cfg.dim, cfg.conc_dim,
                              cfg.heads, cfg.layers, cfg.epochs), ("control", "all", 1024, 128, 8, 1, 3))
        saved = self.root / "model_example_control"
        saved.mkdir()
        (saved / "last.pt").touch()
        (saved / "config.json").write_text(json.dumps({"model_type": "control", "mode": "all"}))
        with patch("models.run.check_environment", return_value="off"), patch(
                "models.training.config.TRAINED_ROOT", self.root), patch(
                "builtins.input", side_effect=["2", "1", "7", "3"]), patch(
                "models.training.engine.train") as training:
            main()
            training.assert_called_once_with(resume=str(saved), additional_epochs=7, batch_size=3)

    def test_head_sizing_defaults_and_command_line(self):
        _, cfg = parse_args(["--heads", "8", "--head-dim", "128"])
        self.assertEqual((cfg.dim, cfg.conc_dim, cfg.head_dim), (1024, 128, 128))
        self.assertEqual(cfg.grad_clip, 3.)
        self.assertEqual((cfg.n_control, cfg.n_target), (512, 128))
        _, single = parse_args(["--model", "control", "--heads", "4", "--head-dim", "16",
                                "--file", str(self.data / "plate1_A.h5ad")], single=True)
        self.assertEqual((single.dim, single.conc_dim, single.grad_clip), (64, 16, 3.))
        cfg = TrainingConfig(heads=2, head_dim=8)
        self.assertEqual((cfg.dim, cfg.conc_dim), (16, 8))
        cfg.dim = 32
        with self.assertRaisesRegex(ValueError, "heads \* head_dim"):
            cfg.validate()
        with self.assertRaises(ValueError):
            TrainingConfig(heads=0, head_dim=8)
        wide = TrainingConfig(heads=8, head_dim=150)
        wide.validate()
        self.assertEqual((wide.dim, wide.conc_dim), (1200, 150))
        with self.assertRaisesRegex(ValueError, "divisible by heads"):
            TrainingConfig(dim=1201, heads=8).validate()

    def test_derived_head_shapes_train_and_save_for_both_models(self):
        for model_type in ("control", "full"):
            cfg = self.cfg(model_type=model_type, head_dim=8, mode="single",
                           file=str(self.data / "plate1_A.h5ad"))
            with patch("models.training.checkpoints.TRAINED_ROOT", self.root / "runs"), patch(
                    "models.training.engine.load_chemistry", side_effect=lambda c:
                    FakeChemistry() if c.model_type == "full" else None):
                run = train(cfg)
            _, state = load_checkpoint(run)
            self.assertEqual((state["config"]["heads"], state["config"]["head_dim"],
                              state["config"]["dim"], state["config"]["conc_dim"]), (2, 8, 16, 8))
            self.assertEqual(tuple(state["model"]["conc_emb.weight"].shape), (8, 16))
            self.assertEqual(state["config"]["grad_clip"], 3.)

    def test_legacy_resume_preserves_shape_and_uses_clipping_three(self):
        with patch("models.training.checkpoints.TRAINED_ROOT", self.root / "runs"):
            run = train(self.cfg(grad_clip=1.))
            train(resume=str(run), additional_epochs=1)
        _, state = load_checkpoint(run)
        self.assertEqual((state["config"]["dim"], state["config"]["conc_dim"]), (16, 4))
        self.assertEqual(state["config"]["grad_clip"], 3.)
        self.assertEqual(state["progress"]["epoch"], 2)

    def test_single_rejects_excluded_plate_and_cli_resume_uses_saved_config(self):
        cfg = self.cfg(mode="single", file=str(self.write_file(13, "A")))
        with self.assertRaisesRegex(ValueError, "plate1..plate12"):
            make_plan(cfg)
        args, cfg = parse_args(["--resume", "some_run", "--epochs", "3"])
        self.assertEqual(args.epochs, 3)
        self.assertIsNone(cfg)
        with self.assertRaises(SystemExit):
            parse_args(["--resume", "some_run", "--head-dim", "64"])


if __name__ == "__main__":
    unittest.main()

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import test_training as fixtures
from models.training.config import TrainingConfig
from models.training.data import DataPlan, epoch_units
from models.training.engine import train
from models.training.checkpoints import load_checkpoint


class ClassRotationTests(unittest.TestCase):
    def plan(self):
        samples = [f"w{i}" for i in range(16)]
        record = dict(path="plate1_A.h5ad", plate=1, cell_line="A", target_samples=samples,
                      target_drugs={s:f"d{i}" for i,s in enumerate(samples)}, control_samples=["c1","c2"])
        return DataPlan([record, {**record, "path":"plate2_A.h5ad", "plate":2},
                         {**record, "path":"plate1_V.h5ad", "cell_line":"V"}],
                        ["A"],["V"],4,["g"]*4,{f"d{i}":i//2+1 for i in range(16)})

    def test_all_files_distinct_classes_controls_and_four_epoch_coverage(self):
        plan=self.plan();cfg=TrainingConfig();rng=np.random.default_rng(8);state={};seen={}
        for _ in range(4):
            units=epoch_units(plan,cfg,rng,state)
            self.assertEqual(len(units),2)
            for u in units:
                self.assertEqual(u["cell_line"],"A")
                self.assertEqual(len(set(u["chemical_classes"])),2)
                self.assertEqual(set(u["selected_controls"]),{"c1","c2"})
                seen.setdefault(u["path"],set()).update(u["chemical_classes"])
        self.assertTrue(all(v==set(range(1,9)) for v in seen.values()))
        # Two drugs per class: the next cycle rotates to the other member.
        for _ in range(4):epoch_units(plan,cfg,rng,state)
        for counters in state["files"].values():
            self.assertEqual(set(counters["wells"].values()),{1})
            self.assertEqual(len(counters["wells"]),16)

    def test_sparse_files_and_larger_or_short_batches(self):
        plan=self.plan();plan.files=plan.files[:1];plan.files[0]["target_samples"]=["w0","w1","w2"]
        for b in (1,2,4):
            cfg=TrainingConfig(batch_size=b)
            u=epoch_units(plan,cfg,np.random.default_rng(0),{})[0]
            self.assertEqual(len(u["samples"]),min(b,3))
            self.assertEqual(len(set(u["samples"])),len(u["samples"]))
            if b>1:self.assertEqual(len(set(u["chemical_classes"])),2)
            self.assertEqual(len(u["selected_controls"]),len(u["samples"]))

    def test_schedule_and_counters_resume_exactly(self):
        plan=self.plan();cfg=TrainingConfig();rng=np.random.default_rng(7);state={}
        epoch_units(plan,cfg,rng,state)
        restored=json.loads(json.dumps(state));rng2=np.random.default_rng();rng2.bit_generator.state=copy.deepcopy(rng.bit_generator.state)
        self.assertEqual(epoch_units(plan,cfg,rng,state),epoch_units(plan,cfg,rng2,restored))
        self.assertEqual(state,restored)

    def test_schema_eleven_keeps_legacy_sampler(self):
        config=TrainingConfig().to_dict();config.pop("sampling_strategy")
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"last.pt"
            torch.save(dict(schema_version=11,config=config),path)
            _,state=load_checkpoint(path)
            self.assertEqual(state["config"]["sampling_strategy"],"legacy")

    def test_capped_schedule_does_not_count_unselected_files(self):
        state={};units=epoch_units(self.plan(),TrainingConfig(max_units_per_epoch=1),np.random.default_rng(0),state)
        self.assertEqual(len(units),1);self.assertEqual(len(state["files"]),1)


class ClassRotationPipelineTests(unittest.TestCase):
    def test_real_pipeline_exact_resume_and_logged_pairing(self):
        fixture=fixtures.PipelineTests();fixture.setUp()
        try:
            mapping=fixture.root/"classes.json"
            mapping.write_text(json.dumps({"classes":{"drugA":1,"drugB":8}}))
            cfg=fixture.cfg(model_type="full",sampling_strategy="class_rotation",chemical_classes=str(mapping),
                            batch_size=2,accumulate_batches=False,epochs=1,val_cell_lines=1)
            with patch("models.training.checkpoints.TRAINED_ROOT",fixture.root/"runs"), patch("models.training.engine.load_chemistry",return_value=fixtures.FakeChemistry()):
                run=train(copy.deepcopy(cfg));_,reference=load_checkpoint(run)
                interrupted=copy.deepcopy(cfg);interrupted.max_steps=1
                partial=train(interrupted)
                train(resume=str(partial),additional_epochs=1)
                _,resumed=load_checkpoint(partial)
            self.assertEqual(reference["progress"]["optimizer_step"],2)
            self.assertEqual(reference["progress"]["sampler_state"],resumed["progress"]["sampler_state"])
            for key,value in reference["model"].items():
                torch.testing.assert_close(value,resumed["model"][key],rtol=0,atol=0)
            with (run/"log.csv").open() as f:rows=list(csv.DictReader(f))
            for row in rows:
                details=json.loads(row["batch_samples"])
                self.assertEqual({d["control_sample"] for d in details},{"dmso1","dmso2"})
                self.assertEqual({d["chemical_class"] for d in details},{1,8})
                self.assertEqual(row["accumulation_count"],"1")
        finally:fixture.tearDown()

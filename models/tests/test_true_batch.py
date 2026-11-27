import copy,csv,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import test_training as fixtures
from models.training.config import TrainingConfig,parse_args
from models.training.modeling import create_model,predict_batch,loss_for,losses_for_batch
from models.training.checkpoints import load_checkpoint
from models.training.engine import train
from models.training.capacity import run_trial
from models.training.metrics import PassLog

class VariableTokens:
    dim=5
    poison=False
    def collate(self,drugs,doses,device=None,mode='all'):
        chem=torch.full((len(drugs),5,6),10000. if self.poison else 0.,device=device)
        mask=torch.zeros(len(drugs),1,1,5,dtype=torch.bool,device=device)
        for i,(drug,dose) in enumerate(zip(drugs,doses)):
            n=2 if drug=='A' else 5
            chem[i,:n]=torch.arange(n*6,device=device).reshape(n,6)/10
            chem[i,:n,-1]=dose;mask[i,0,0,:n]=True
        return chem,mask

class TrueBatchUnitTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);torch.set_num_threads(1)
        self.cfg=TrainingConfig(heads=2,head_dim=8,layers=2,dropout=0,amp='off',query_chunk=7,readout_chunk=3)
        self.ex=[SimpleNamespace(control=torch.rand(24,16),coords=torch.rand(8,2),target=torch.rand(8,24),drug=d,concentration=.05+i) for i,d in enumerate(['A','B','A'])]
        self.store=VariableTokens();self.device=torch.device('cpu')
    def test_batched_output_and_all_gradients_match_accumulation_with_padding(self):
        model=create_model(self.cfg,24,6).eval();reference=copy.deepcopy(model)
        predicted=predict_batch(model,self.ex[:2],self.cfg,self.device,self.store)
        self.assertEqual(predicted.shape,(2,8,24))
        individual=torch.cat([predict_batch(model,[ex],self.cfg,self.device,self.store) for ex in self.ex[:2]])
        torch.testing.assert_close(predicted,individual,atol=1e-6,rtol=1e-5)
        losses_for_batch(model,self.ex[:2],self.cfg,self.device,self.store).mean().backward()
        for ex in self.ex[:2]:(loss_for(reference,ex,self.cfg,self.device,self.store)/2).backward()
        for (name,p),(_,q) in zip(model.named_parameters(),reference.named_parameters()):
            self.assertIsNotNone(p.grad,name);torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4)
        self.store.poison=True
        torch.testing.assert_close(predicted,predict_batch(model,self.ex[:2],self.cfg,self.device,self.store),atol=1e-6,rtol=1e-5)
    def test_layer_checkpointing_matches_outputs_gradients_and_dropout_rng(self):
        self.cfg.dropout=.1
        model=create_model(self.cfg,24,6).train();reference=copy.deepcopy(model)
        reference.use_checkpoint_layers=False
        torch.manual_seed(43)
        loss=losses_for_batch(model,self.ex[:2],self.cfg,self.device,self.store).mean()
        loss.backward();rng_after=torch.get_rng_state().clone()
        torch.manual_seed(43)
        expected=losses_for_batch(reference,self.ex[:2],self.cfg,self.device,self.store).mean()
        expected.backward()
        torch.testing.assert_close(loss,expected,rtol=0,atol=0)
        torch.testing.assert_close(torch.get_rng_state(),rng_after,rtol=0,atol=0)
        for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4)
    def test_control_batch_is_independent(self):
        self.cfg.model_type='control';model=create_model(self.cfg,24).eval()
        y=predict_batch(model,self.ex[:2],self.cfg,self.device)
        separate=torch.cat([predict_batch(model,[e],self.cfg,self.device) for e in self.ex[:2]])
        torch.testing.assert_close(y,separate,atol=1e-6,rtol=1e-5)
    def test_short_last_batch_uses_equal_well_weight(self):
        model=create_model(self.cfg,24,6).eval();reference=copy.deepcopy(model)
        for batch in (self.ex[:2],self.ex[2:]):
            (losses_for_batch(model,batch,self.cfg,self.device,self.store).mean()*len(batch)/3).backward()
        for ex in self.ex:(loss_for(reference,ex,self.cfg,self.device,self.store)/3).backward()
        for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4)
    def test_old_schema_preserves_batch_one_and_new_requires_setting(self):
        config=self.cfg.to_dict();config.pop('batch_size');config.pop('accumulate_batches');config.pop('checkpoint_layers')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.pt';torch.save(dict(schema_version=9,config=config),path)
            _,state=load_checkpoint(path);self.assertEqual(state['config']['batch_size'],1);self.assertTrue(state['config']['accumulate_batches'])
            torch.save(dict(schema_version=10,config=config),path)
            with self.assertRaisesRegex(ValueError,'batch_size'):load_checkpoint(path)
        _,cfg=parse_args([]);self.assertEqual(cfg.batch_size,2)
        _,cfg=parse_args(['--batch-size','1']);self.assertEqual(cfg.batch_size,1)
    def test_csv_without_batch_columns_migrates_as_single_well(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'log.csv';log=PassLog(path,['test']);fields=[f for f in log.fields if f not in ('batch_size','batch_samples')];log.close()
            with path.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerow(dict(backward_pass=1,loss=2,grad_total=1,grad_test=1))
            log=PassLog(path,['test'],committed_passes=1);log.close()
            with path.open() as stream:row=next(csv.DictReader(stream))
            self.assertEqual(row['batch_size'],'1');self.assertEqual(row['batch_samples'],'')

class TrueBatchPipelineTests(unittest.TestCase):
    def setUp(self):self.fixture=fixtures.PipelineTests();self.fixture.setUp()
    def tearDown(self):self.fixture.tearDown()
    def test_real_two_drug_forward_logging_and_exact_resume(self):
        cfg=self.fixture.cfg(model_type='full',batch_size=2,accumulate_batches=False,passes_per_cell_line=2)
        with patch('models.training.checkpoints.TRAINED_ROOT',self.fixture.root/'runs'),patch('models.training.engine.load_chemistry',return_value=fixtures.FakeChemistry()):
            shapes=[]
            def factory(*args,**kwargs):
                model=create_model(*args,**kwargs)
                model.register_forward_pre_hook(lambda m,a:shapes.append(a[0].shape[0]))
                return model
            with patch('models.training.engine.create_model',side_effect=factory):run=train(cfg)
            _,reference=load_checkpoint(run)
            interrupted=train(self.fixture.cfg(model_type='full',batch_size=2,accumulate_batches=False,passes_per_cell_line=2,max_steps=1))
            train(resume=str(interrupted),additional_epochs=1);_,resumed=load_checkpoint(interrupted)
        self.assertTrue(shapes);self.assertTrue(all(b==2 for b in shapes))
        self.assertEqual(reference['progress']['optimizer_step'],2);self.assertEqual(reference['progress']['backward_pass'],2)
        for key,value in reference['model'].items():torch.testing.assert_close(value,resumed['model'][key],rtol=0,atol=0)
        rows=list(csv.DictReader((run/'log.csv').open()));self.assertEqual(len(rows),2)
        for row in rows:
            self.assertEqual(row['batch_size'],'2');self.assertEqual(row['accumulation_count'],'1')
            detail=json.loads(row['batch_samples']);self.assertEqual({d['drug'] for d in detail},{'drugA','drugB'})
            for key in ('loss','t_mean_loss','i_mean_loss'):self.assertAlmostEqual(float(row[key]),sum(d[key] for d in detail)/2,places=5)
    def test_selected_B_fills_batches_and_updates_without_accumulation(self):
        import anndata as ad
        import numpy as np
        path=self.fixture.data/'plate1_A.h5ad'
        data=ad.read_h5ad(path);extra=data[12:24].copy()
        extra.obs['sample']=np.repeat(['drug3','drug4'],6)
        extra.obs['drug']=np.repeat(['drugC','drugD'],6)
        ad.concat([data,extra],index_unique='-').write_h5ad(path)
        for size,expected_steps in [(1,2),(3,1)]:
            cfg=self.fixture.cfg(model_type='full',batch_size=size,accumulate_batches=False,mode='all',passes_per_cell_line=1)
            # Use one real selected file while retaining all-file sampling semantics.
            from models.training.data import DataPlan
            plan=DataPlan([dict(path=str(path),plate=1,cell_line='A',target_samples=['drug1','drug2','drug3','drug4'],target_drugs=dict(drug1='drugA',drug2='drugB',drug3='drugC',drug4='drugD'))],['A'],[],16,list(data.var_names))
            with patch('models.training.checkpoints.TRAINED_ROOT',self.fixture.root/'runs'),patch('models.training.engine.make_plan',return_value=plan),patch('models.training.engine.load_chemistry',return_value=fixtures.FakeChemistry()):run=train(cfg)
            _,state=load_checkpoint(run)
            self.assertEqual(state['progress']['optimizer_step'],expected_steps)
            self.assertEqual(state['progress']['backward_pass'],expected_steps)
            with (run/'log.csv').open() as stream:rows=list(csv.DictReader(stream))
            self.assertTrue(all(int(row['batch_size'])==size and row['accumulation_count']=='1' for row in rows))
    def test_resume_batch_override_disables_old_accumulation(self):
        with patch('models.training.checkpoints.TRAINED_ROOT',self.fixture.root/'runs'):
            run=train(self.fixture.cfg(batch_size=1,accumulate_batches=True,passes_per_cell_line=2,max_steps=1))
            saved=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
            saved['config']['max_steps']=0;torch.save(saved,run/'last.pt')
            train(resume=str(run),additional_epochs=1,batch_size=1)
        _,state=load_checkpoint(run)
        self.assertFalse(state['config']['accumulate_batches'])
        self.assertEqual(state['progress']['optimizer_step'],7)
        with (run/'log.csv').open() as stream:rows=list(csv.DictReader(stream))
        self.assertTrue(all(row['accumulation_count']=='1' for row in rows[2:]))
    def test_single_mode_batches_and_capacity_reports_true_forwards(self):
        cfg=self.fixture.cfg(model_type='full',mode='single',file=str(self.fixture.data/'plate1_A.h5ad'),batch_size=2,accumulate_batches=False)
        with patch('models.training.checkpoints.TRAINED_ROOT',self.fixture.root/'runs'),patch('models.training.engine.load_chemistry',return_value=fixtures.FakeChemistry()):run=train(cfg)
        _,state=load_checkpoint(run);self.assertEqual(state['progress']['optimizer_step'],1)
        with patch('models.training.capacity.load_chemistry',return_value=fixtures.FakeChemistry()):result=run_trial(cfg,steps=1,warmup=0)
        self.assertIn(result['status'],('fits','tight','cpu_completed'),result);self.assertEqual(result['batch_size'],2);self.assertEqual(result['micro_batches_per_unit'],1)

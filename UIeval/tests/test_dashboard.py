"""Tests for incremental reads and dashboard interactions; no real run is changed."""
from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from UIeval.data import LiveLogReader, RunReader, discover_runs, filter_frame, x_values

HEADER = ['epoch','optimizer_step','backward_pass','file','plate','cell_line','loss',
          'grad_total','grad_merger','seconds','peak_vram_gb']


def append_row(path, backward, loss, line='A'):
    with path.open('a',newline='',encoding='utf-8') as fh:
        csv.writer(fh).writerow([1,backward,backward,'plate1_A.h5ad',1,line,loss,2.,.5,.2,1.2])


def make_run(root,name='model_A_control',loss=4.):
    run=root/name;run.mkdir(parents=True)
    with (run/'log.csv').open('w',newline='',encoding='utf-8') as fh:csv.writer(fh).writerow(HEADER)
    append_row(run/'log.csv',1,loss)
    config=dict(model_type='control',mode='all',epochs=2,dim=64,heads=4,head_dim=16,
                conc_dim=16,layers=1,normalization='layernorm',gene_statistics=True,grad_clip=3.,amp='bf16',lr=.0001)
    (run/'config.json').write_text(json.dumps(config),encoding='utf-8')
    (run/'info.json').write_text(json.dumps(dict(status='running',config=config,epochs=[],
        model=dict(n_genes=16,parameter_total=1000,parameters=dict(merger=100)))),encoding='utf-8')
    return run


class LiveReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='uie_eval_')
        self.root=Path(self.temporary.name)

    def tearDown(self):self.temporary.cleanup()

    def test_incremental_append_and_partial_last_record(self):
        run=make_run(self.root);path=run/'log.csv';reader=LiveLogReader()
        self.assertEqual(len(reader.poll(path).frame),1)
        with path.open('ab') as fh:fh.write(b'1,2,2,plate1_A.h5ad,1,B,3.25,2,0.5,0.2,')
        snapshot=reader.poll(path)
        self.assertTrue(snapshot.pending);self.assertEqual(len(snapshot.frame),1)
        with path.open('ab') as fh:fh.write(b'1.4\n')
        snapshot=reader.poll(path)
        self.assertFalse(snapshot.pending);self.assertEqual(len(snapshot.frame),2)
        self.assertEqual(snapshot.frame.iloc[-1]['loss'],3.25)
        self.assertEqual(len(reader.poll(path).frame),2)

    def test_atomic_replace_truncate_and_rewrite_past_old_offset(self):
        run=make_run(self.root);path=run/'log.csv';reader=LiveLogReader()
        append_row(path,2,3.)
        reader.poll(path)
        replacement=run/'replacement.csv'
        with replacement.open('w',newline='',encoding='utf-8') as fh:csv.writer(fh).writerow(HEADER)
        append_row(replacement,1,9.)
        os.replace(replacement,path)
        self.assertEqual(reader.poll(path).frame.loss.tolist(),[9.])
        with path.open('w',newline='',encoding='utf-8') as fh:csv.writer(fh).writerow(HEADER)
        for i in range(1,6):append_row(path,i,10.+i)
        self.assertEqual(reader.poll(path).frame.loss.tolist(),[11.,12.,13.,14.,15.])
        path.write_bytes(b'')
        self.assertTrue(reader.poll(path).frame.empty)

    def test_malformed_rows_and_missing_file(self):
        run=make_run(self.root);path=run/'log.csv';reader=LiveLogReader()
        with path.open('ab') as fh:fh.write(b'bad,row\n1,2,2,plate1_A.h5ad,1,B,nan,2,0.5,0.2,1\n')
        snapshot=reader.poll(path)
        self.assertEqual(len(snapshot.frame),1)
        self.assertTrue(any('2 malformed' in note for note in snapshot.warnings))
        path.unlink();self.assertTrue(reader.poll(path).frame.empty)

    def test_json_last_good_snapshot_and_run_discovery(self):
        run=make_run(self.root);reader=RunReader()
        _,info,cfg,_=reader.poll(run)
        self.assertEqual(info['status'],'running');self.assertEqual(cfg['grad_clip'],3.)
        (run/'info.json').write_text('{',encoding='utf-8')
        _,info,_,notes=reader.poll(run)
        self.assertEqual(info['status'],'running');self.assertTrue(notes)
        self.assertEqual(discover_runs(self.root),[run])
        self.assertEqual(discover_runs(self.root/'missing'),[])

    def test_filters_keep_original_horizontal_axis(self):
        frame=pd.DataFrame(dict(epoch=[1,1,2],plate=[1,2,1],cell_line=['A','B','A'],loss=[3.,2.,1.]))
        view=filter_frame(frame,epochs=[2],lines=['A'])
        values,label=x_values(view)
        self.assertEqual(values.tolist(),[3]);self.assertEqual(label,'Log row')


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='uie_app_')
        self.root=Path(self.temporary.name)
        self.environment=patch.dict(os.environ,{'UIEVAL_RUNS_ROOT':str(self.root)})
        self.environment.start()
        self.app_path=str(Path(__file__).resolve().parents[1]/'app.py')

    def tearDown(self):
        self.environment.stop();self.temporary.cleanup()

    def test_empty_directory_and_waiting_run(self):
        app=AppTest.from_file(self.app_path,default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('No runs found' in info.value for info in app.info))
        run=make_run(self.root);(run/'log.csv').unlink()
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any('first complete' in info.value for info in app.info))

    def test_append_filters_pause_run_selection_and_comparison(self):
        run=make_run(self.root,loss=4.)
        make_run(self.root,'model_B_control',loss=7.)
        app=AppTest.from_file(self.app_path,default_timeout=20).run()
        self.assertFalse(app.exception)
        app.selectbox(key='selected_run').select(run.name).run()
        self.assertEqual(app.metric[0].value,'4.0000')
        append_row(run/'log.csv',2,3.25,'B')
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value,'3.2500')
        self.assertEqual(app.metric[2].value,'2')
        app.multiselect(key=run.name+':lines').select('A').run()
        self.assertFalse(app.exception)
        self.assertTrue(any('1 of 2 complete records' in cap.value for cap in app.caption))
        app.multiselect(key='comparison_runs').select('model_B_control').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.selectbox(key='comparison_metric').options),4)
        app.toggle(key='live_updates').set_value(False).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Paused' in cap.value for cap in app.caption))
        self.assertEqual(len(app.tabs),6)


if __name__=='__main__':unittest.main()

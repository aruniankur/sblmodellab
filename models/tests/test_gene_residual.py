"""Residual placement, identity gradient path, and legacy forward compatibility."""
import tempfile
import unittest
from pathlib import Path
import torch
from models.control_model import ControlPerturbationModel
from models.model import PerturbationModel
from models.training.checkpoints import load_checkpoint

class GeneResidualTests(unittest.TestCase):
    def test_identity_path_precedes_norm_in_both_models(self):
        for cls in (ControlPerturbationModel, PerturbationModel):
            kwargs=dict(dim=16, conc_dim=4, num_heads=2, num_layers=1, num_of_genes=8, dropout=0.)
            if cls is PerturbationModel: kwargs['chem_dim']=5
            model=cls(**kwargs).eval()
            with torch.no_grad():
                model.merger.weight.zero_(); model.merger.bias.zero_()
            captured=[]
            handle=model.merger_norm.register_forward_pre_hook(lambda m,args: captured.append(args[0]))
            control=torch.rand(1,8,6);coords=torch.rand(1,3,2)
            args=(control,coords) if cls is ControlPerturbationModel else (control,torch.rand(1,2,5),coords)
            model(*args)
            torch.testing.assert_close(captured[0], model.gene_embedding.weight.unsqueeze(0))
            captured[0].sum().backward()
            torch.testing.assert_close(model.gene_embedding.weight.grad,torch.ones_like(model.gene_embedding.weight))
            model.use_gene_residual=False
            model(*args)
            torch.testing.assert_close(captured[-1],torch.zeros_like(captured[-1]))
            handle.remove()

    def test_old_checkpoint_disables_residual_and_new_requires_setting(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.pt'
            torch.save({'schema_version':3,'config':{'normalization':'layernorm','gene_statistics':True}},path)
            _,state=load_checkpoint(path)
            self.assertFalse(state['config']['gene_residual'])
            torch.save({'schema_version':4,'config':{'normalization':'layernorm','gene_statistics':True}},path)
            with self.assertRaisesRegex(ValueError,'gene-residual'):load_checkpoint(path)

if __name__=='__main__':unittest.main()

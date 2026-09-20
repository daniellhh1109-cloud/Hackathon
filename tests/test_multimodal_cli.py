import argparse
import sys
from types import SimpleNamespace
import pytest
import torch
from scripts import run_multimodal as cli
from src.training.multimodal import collate_multimodal

@pytest.mark.parametrize('spec',['missingcolon',':name','module:','module:name:extra','bad-module:name'])
def test_factory_format(spec,capsys):
    with pytest.raises(SystemExit) as exc: cli.load_factory(spec,2021,argparse.ArgumentParser())
    assert exc.value.code==2
    assert 'module:name' in capsys.readouterr().err

@pytest.mark.parametrize('value,message',[(42,'not callable'),(lambda: {},'year keyword'),(lambda year, other: {},'year keyword'),(lambda year: [],'component dictionary')])
def test_factory_contract(monkeypatch,capsys,value,message):
    monkeypatch.setattr(cli.importlib,'import_module',lambda _:SimpleNamespace(build=value))
    with pytest.raises(SystemExit): cli.load_factory('fixture:build',2021,argparse.ArgumentParser())
    assert message in capsys.readouterr().err

@pytest.mark.parametrize('missing_module',[True,False])
def test_factory_lookup(monkeypatch,capsys,missing_module):
    def lookup(_):
        if missing_module: raise ModuleNotFoundError('fixture is unavailable')
        return SimpleNamespace()
    monkeypatch.setattr(cli.importlib,'import_module',lookup)
    with pytest.raises(SystemExit): cli.load_factory('fixture:build',2021,argparse.ArgumentParser())
    assert 'Failed to load factory' in capsys.readouterr().err

def test_factory_success_and_internal_bug(monkeypatch):
    monkeypatch.setattr(cli.importlib,'import_module',lambda _:SimpleNamespace(build=lambda *,year:{'year':year}))
    assert cli.load_factory('fixture:build',2022,argparse.ArgumentParser())=={'year':2022}
    def broken(*,year): raise TypeError('internal factory bug')
    monkeypatch.setattr(cli.importlib,'import_module',lambda _:SimpleNamespace(build=broken))
    with pytest.raises(TypeError,match='internal factory bug'): cli.load_factory('fixture:build',2021,argparse.ArgumentParser())

def test_smoke_year_message(monkeypatch,capsys,tmp_path):
    monkeypatch.setattr(sys,'argv',['run_multimodal','--smoke','--year','2022','--output-dir',str(tmp_path/'unused')])
    with pytest.raises(SystemExit): cli.main()
    assert 'fixed 2021 train/validation split' in capsys.readouterr().err
    assert not (tmp_path/'unused').exists()

def test_all_empty_batch_backward_and_reload_device():
    rows=cli.smoke_components()['train_dataset'][::4]
    assert all(row['filings'].shape==(6,0,384) for row in rows)
    batch=collate_multimodal(rows)
    assert batch['filings'].shape==(16,6,1,384)
    assert not batch['filing_mask'].any()
    model=cli.SmokeMultimodal()
    out=model(batch['quant'],batch['filings'],batch['filing_mask'],batch['filing_counts'])
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(model.head.weight.grad).all()
    class Probe(cli.SmokeMultimodal):
        def to(self,device):
            self.requested_device=device
            return super().to(device)
    first,second=Probe(),Probe()
    second.load_state_dict(first.state_dict())
    actual,expected=cli.compare_reloaded_models(first,second,batch)
    assert first.requested_device==second.requested_device=='cpu'
    assert not first.training and not second.training
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)

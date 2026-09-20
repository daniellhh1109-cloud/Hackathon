"""Behavioral acceptance for A's trainer, independent of unfinished B/C modules."""
import csv
from dataclasses import replace

import pytest
import torch
from torch import nn

from scripts.run_member_a import smoke_components, SmokeRegressor
from src.training.trainer import (TrainingConfig, fit, load_checkpoint, _run_epoch,
                                  annual_bounds, read_config)
from torch.utils.data import DataLoader


def run(tmp_path, **overrides):
    components = smoke_components()
    components.update(overrides)
    return fit(**components, config=TrainingConfig(lr=0.1, batch_size=16, max_epochs=12,
               early_stopping_patience=3), target_year=2021, output_dir=tmp_path), components


def test_learning_checkpoint_and_reproducibility(tmp_path):
    (model, summary), components = run(tmp_path / 'first')
    (_, second), _ = run(tmp_path / 'second')
    assert summary['best_validation_loss'] == second['best_validation_loss']
    rows = list(csv.DictReader((tmp_path / 'first/history.csv').open()))
    assert float(rows[-1]['train_loss']) < float(rows[0]['train_loss'])
    assert summary['best_validation_loss'] == min(float(r['validation_loss']) for r in rows)
    loaded = SmokeRegressor()
    checkpoint = load_checkpoint(tmp_path / 'first/best.pt', loaded,
        expected_feature_names=components['feature_names'], expected_model_metadata=components['model_metadata'])
    assert checkpoint['epoch'] == summary['best_epoch']
    assert not loaded.training and not model.training
    quant = components['validation_dataset'].quant[:5]
    torch.testing.assert_close(loaded(quant), model(quant), rtol=0, atol=0)
    assert checkpoint['data_audit']['train']['target_month_range'] == ['2018-12', '2018-12']
    with pytest.raises(ValueError, match='feature order'):
        load_checkpoint(tmp_path / 'first/best.pt', loaded,
            expected_feature_names=components['feature_names'][::-1], expected_model_metadata=components['model_metadata'])
    with pytest.raises(ValueError, match='architecture'):
        load_checkpoint(tmp_path / 'first/best.pt', loaded,
            expected_feature_names=components['feature_names'], expected_model_metadata={'name': 'different'})
    with pytest.raises(FileExistsError):
        run(tmp_path / 'first')


def test_early_stop_restores_best_not_last(tmp_path, monkeypatch):
    from src.training import trainer
    values = iter([0.5, 0.2, 0.3, 0.4])
    def controlled_epoch(model, loader, criterion, device, optimizer=None):
        if optimizer is not None:
            with torch.no_grad():
                model.head.bias.add_(1)
            return 1.0
        return next(values)
    monkeypatch.setattr(trainer, '_run_epoch', controlled_epoch)
    components = smoke_components()
    model, summary = fit(**components, config=TrainingConfig(max_epochs=20, early_stopping_patience=2),
                         target_year=2021, output_dir=tmp_path / 'stop')
    assert summary['epochs_run'] == 4
    assert summary['best_epoch'] == 2
    assert summary['stop_reason'] == 'early_stopping'
    saved = torch.load(tmp_path / 'stop/best.pt', weights_only=True)
    torch.testing.assert_close(model.head.bias, saved['model_state_dict']['head.bias'])


def test_validation_eval_no_grad_and_sample_weighting():
    class Probe(nn.Module):
        def forward(self, quant):
            assert not self.training
            assert not torch.is_grad_enabled()
            return torch.zeros(len(quant), 1)
    rows = [{'quant': torch.zeros(12,147), 'target': torch.tensor(v)} for v in [0., 0., 3.]]
    loss = _run_epoch(Probe(), DataLoader(rows, batch_size=2), nn.HuberLoss(), 'cpu')
    assert loss == pytest.approx(2.5 / 3)  # last, smaller batch must not count equally


@pytest.mark.parametrize('fault,match', [
    ('future', 'outside annual split'), ('alignment', 'next calendar month'),
    ('nan', 'finite scalar'), ('shape', 'finite \\[12,147\\]'), ('duplicate', 'duplicate')])
def test_bad_data_rejected_before_artifacts(tmp_path, fault, match):
    components = smoke_components()
    rows = [components['train_dataset'][i] for i in range(3)]
    if fault == 'future':
        rows[0].update(target_month='2021-01', quant_end_month='2020-12')
    elif fault == 'alignment':
        rows[0]['quant_end_month'] = '2018-10'
    elif fault == 'nan':
        rows[0]['target'] = torch.tensor(float('nan'))
    elif fault == 'shape':
        rows[0]['quant'] = torch.zeros(11, 147)
    else:
        rows[1] = rows[0]
    components['train_dataset'] = rows
    with pytest.raises(ValueError, match=match):
        fit(**components, config=TrainingConfig(), target_year=2021, output_dir=tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_prevent_broadcast_and_nonfinite_prediction():
    class Invalid(nn.Module):
        def forward(self, quant):
            return torch.zeros(len(quant), len(quant))
    loader = DataLoader(smoke_components()['validation_dataset'], batch_size=4)
    with pytest.raises(ValueError, match='shape'):
        _run_epoch(Invalid(), loader, nn.HuberLoss(), 'cpu')
    class Nonfinite(nn.Module):
        def forward(self, quant):
            return torch.full((len(quant),), float('nan'))
    with pytest.raises(ValueError, match='non-finite prediction'):
        _run_epoch(Nonfinite(), loader, nn.HuberLoss(), 'cpu')


@pytest.mark.parametrize('year', range(2021, 2027))
def test_annual_boundaries(year):
    bounds = annual_bounds(year)
    assert bounds['train'] == ['2015-01', f'{year-3}-12']
    assert bounds['validation'] == [f'{year-2}-01', f'{year-1}-12']
    assert bounds['test'][1] == ('2026-08' if year == 2026 else f'{year}-12')


@pytest.mark.parametrize('change', [dict(lr=0), dict(huber_delta=-1), dict(batch_size=0),
    dict(max_epochs=0), dict(early_stopping_patience=0), dict(min_delta=float('nan'))])
def test_invalid_config(change):
    with pytest.raises(ValueError):
        replace(TrainingConfig(), **change)


def test_config_and_feature_leakage(tmp_path):
    assert read_config('configs/member_a_week2.yaml').batch_size == 512
    components = smoke_components()
    components['feature_names'][0] = 'ret_exc_lead1m'
    with pytest.raises(ValueError, match='label/identifier'):
        fit(**components, config=TrainingConfig(), target_year=2021, output_dir=tmp_path / 'bad')


@pytest.mark.parametrize('field', ['model_metadata', 'preprocessing_metadata', 'extra_metadata'])
@pytest.mark.parametrize('fault', ['nan', 'infinity', 'unsupported', 'circular'])
def test_invalid_metadata_identifies_argument_before_creating_output(tmp_path, field, fault):
    value = {'nested': {'value': float('nan')}}
    if fault == 'infinity':
        value = {'nested': {'value': float('inf')}}
    elif fault == 'unsupported':
        value = {'nested': {'value': object()}}
    elif fault == 'circular':
        value = {}
        value['self'] = value
    output = tmp_path / 'invalid_metadata'
    with pytest.raises(ValueError, match=field + ' must be JSON-serializable') as error:
        run(output, **{field: value})
    expected_cause = TypeError if fault == 'unsupported' else ValueError
    assert isinstance(error.value.__cause__, expected_cause)
    assert not output.exists()

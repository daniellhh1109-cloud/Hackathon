from copy import deepcopy
import pytest
import torch
from torch import nn
from scripts.run_multimodal import SmokeMultimodal, smoke_components
from src.training.multimodal import (validate_provenance, validate_frozen_encoder,
    collate_multimodal, forward_multimodal, load_multimodal_checkpoint)


def test_provenance_returns_validated_mapping():
    metadata = smoke_components()['text_metadata']
    assert validate_provenance(metadata) is metadata
    for bad in (None, [], 'invalid'):
        with pytest.raises(ValueError, match='mapping'):
            validate_provenance(bad)


def test_empty_batch_rejected():
    with pytest.raises(ValueError, match='no samples'):
        collate_multimodal([])


def test_missing_and_inconsistent_quant_fields():
    row = smoke_components()['validation_dataset'][1]
    missing = deepcopy(row); missing.pop('quant')
    with pytest.raises(ValueError, match='missing quant'):
        collate_multimodal([missing])
    missing = deepcopy(row); missing.pop('target')
    for rows in ([row, missing], [missing, row]):
        with pytest.raises(ValueError, match='inconsistent non-text'):
            collate_multimodal(rows)
    row['quant'] = torch.zeros(11,147)
    with pytest.raises(ValueError, match='quant must be finite'):
        collate_multimodal([row])


def test_inference_batch_does_not_require_target():
    rows = smoke_components()['validation_dataset'][:4]
    rows = [{k:v for k,v in row.items() if k != 'target'} for row in rows]
    batch = collate_multimodal(rows)
    assert 'target' not in batch
    assert torch.isfinite(forward_multimodal(SmokeMultimodal().eval(), batch, 'cpu')).all()


def test_bad_sample_is_not_silently_dropped():
    row = smoke_components()['validation_dataset'][1]
    for bad in (None, {}, {'quant':torch.zeros(12,147)}):
        with pytest.raises(ValueError):
            collate_multimodal([row, bad])


def test_freeze_check_is_explicit_and_runs_before_checkpoint_io(monkeypatch):
    model=SmokeMultimodal(); model.text_encoder.requires_grad_(True)
    def forbidden_load(*args,**kwargs):
        pytest.fail('checkpoint must not be read before model validation')
    monkeypatch.setattr(torch,'load',forbidden_load)
    with pytest.raises(ValueError,match='must be frozen'):
        load_multimodal_checkpoint('unused',model,
            expected_text_metadata=smoke_components()['text_metadata'],
            expected_feature_names=smoke_components()['feature_names'],
            expected_model_metadata=smoke_components()['model_metadata'])
    assert all(p.requires_grad for p in model.text_encoder.parameters())
    model.text_encoder=nn.Identity()
    validate_frozen_encoder(model)


def test_late_unfreeze_is_detected():
    model=SmokeMultimodal()
    validate_frozen_encoder(model)
    model.text_encoder.requires_grad_(True)
    batch=collate_multimodal(smoke_components()['validation_dataset'][:2])
    with pytest.raises(ValueError,match='must be frozen'):
        forward_multimodal(model,batch,'cpu')


@pytest.mark.parametrize('missing', ['expected_feature_names', 'expected_model_metadata'])
def test_checkpoint_required_arguments_fail_before_io(monkeypatch, missing):
    components = smoke_components()
    arguments = {
        'expected_text_metadata': components['text_metadata'],
        'expected_feature_names': components['feature_names'],
        'expected_model_metadata': components['model_metadata'],
    }
    del arguments[missing]
    def forbidden_load(*args, **kwargs):
        pytest.fail('missing arguments must fail before checkpoint I/O')
    monkeypatch.setattr(torch, 'load', forbidden_load)
    with pytest.raises(TypeError, match=missing):
        load_multimodal_checkpoint('unused', SmokeMultimodal(), **arguments)


@pytest.mark.parametrize('metadata_first', [False, True])
def test_extra_metadata_rejected_before_default_collate(monkeypatch, metadata_first):
    from src.training import multimodal
    plain = smoke_components()['validation_dataset'][1]
    enriched = dict(plain, metadata={'source': 'fixture'})
    rows = [enriched, plain] if metadata_first else [plain, enriched]

    def forbidden_collate(*args, **kwargs):
        pytest.fail('inconsistent top-level fields must be rejected before default_collate')

    monkeypatch.setattr(multimodal, 'default_collate', forbidden_collate)
    with pytest.raises(ValueError, match='sample 1: inconsistent non-text fields'):
        multimodal.collate_multimodal(rows)

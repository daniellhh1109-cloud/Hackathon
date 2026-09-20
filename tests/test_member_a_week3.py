from copy import deepcopy
import csv
import pytest
import torch
from scripts.run_multimodal import smoke_components, SmokeMultimodal
from src.training.multimodal import (fit_multimodal,collate_multimodal,load_multimodal_checkpoint,forward_multimodal)
from src.training.trainer import TrainingConfig


def test_training_reload_and_frozen_parameters(tmp_path):
    torch.set_num_threads(2)
    c=smoke_components()
    config=TrainingConfig(lr=.03,batch_size=16,max_epochs=30,early_stopping_patience=5)
    model,summary=fit_multimodal(**c,config=config,target_year=2021,output_dir=tmp_path/'run')
    restored=SmokeMultimodal()
    saved=load_multimodal_checkpoint(tmp_path/'run/best.pt',restored,expected_text_metadata=c['text_metadata'],
         expected_feature_names=c['feature_names'],expected_model_metadata=c['model_metadata'])
    assert all(not name.startswith('text_encoder.') for name in saved['trainable_parameter_names'])
    torch.manual_seed(config.seed); initial=SmokeMultimodal()
    for p,q in zip(initial.text_encoder.parameters(),model.text_encoder.parameters()):
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    batch=collate_multimodal(c['validation_dataset'][:8])
    with torch.no_grad():
        torch.testing.assert_close(forward_multimodal(model,batch,'cpu'),forward_multimodal(restored,batch,'cpu'),rtol=0,atol=0)
    history=list(csv.DictReader((tmp_path/'run/history.csv').open()))
    assert float(history[-1]['train_loss'])<float(history[0]['train_loss'])
    assert summary['best_validation_loss']==min(float(r['validation_loss']) for r in history)
    bad=deepcopy(c['text_metadata']);bad['cache_version']='other'
    with pytest.raises(ValueError,match='provenance mismatch'):
        load_multimodal_checkpoint(tmp_path/'run/best.pt',restored,expected_text_metadata=bad,
         expected_feature_names=c['feature_names'],expected_model_metadata=c['model_metadata'])
    model.train();forward_multimodal(model,batch,'cpu')
    assert not model.text_encoder.training


def test_padding_and_all_empty():
    c=smoke_components(); model=SmokeMultimodal().eval()
    row=c['validation_dataset'][1]
    single=collate_multimodal([row]);mixed=collate_multimodal([row,c['validation_dataset'][3]])
    torch.testing.assert_close(forward_multimodal(model,single,'cpu')[0],forward_multimodal(model,mixed,'cpu')[0])
    empty=collate_multimodal([c['validation_dataset'][0]])
    assert empty['filings'].shape==(1,6,1,384)
    assert torch.isfinite(forward_multimodal(model,empty,'cpu')).all()


@pytest.mark.parametrize('fault',['counts','mask','nan','metadata','encoder','future'])
def test_reject_invalid_inputs(tmp_path,fault):
    c=smoke_components()
    row=c['train_dataset'][1]
    if fault=='counts': row['filing_counts']+=1
    if fault=='mask': row['filing_mask']=row['filing_mask'].float()
    if fault=='nan': row['filings'][0,0,0]=float('nan')
    if fault=='metadata': c['text_metadata'].pop('encoder_revision')
    if fault=='future': row.update(target_month='2021-01',quant_end_month='2020-12')
    if fault=='encoder':
        def factory():
            model=SmokeMultimodal();model.text_encoder.requires_grad_(True);return model
        c['model_factory']=factory
    with pytest.raises(ValueError):
        fit_multimodal(**c,config=TrainingConfig(max_epochs=1),target_year=2021,output_dir=tmp_path/'bad')

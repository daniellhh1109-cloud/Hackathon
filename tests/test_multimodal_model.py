from dataclasses import replace
import pytest
import torch
from src.models.text_attention import EventMemory, masked_softmax
from src.models.multimodal_model import MultimodalRegressor, MultimodalConfig, read_multimodal_config, model_metadata


def inputs(batch=3,k=4):
    gen=torch.Generator().manual_seed(44)
    quant=torch.randn(batch,12,147,generator=gen)
    filings=torch.randn(batch,6,k,384,generator=gen)
    mask=torch.rand(batch,6,k,generator=gen)>.45
    mask[0]=False
    if batch>1 and k:mask[1,0,0]=True
    return dict(quant=quant,filings=filings,filing_mask=mask,filing_counts=mask.sum(-1))


@pytest.mark.parametrize('batch,k',[(1,0),(1,1),(3,4)])
@pytest.mark.parametrize('gate_mode',['scalar','vector'])
def test_shapes_empty_rows_and_gradients(batch,k,gate_mode):
    torch.set_num_threads(2)
    model=MultimodalRegressor(MultimodalConfig(gate_mode=gate_mode))
    args=inputs(batch,k)
    y,d=model(**args,return_diagnostics=True)
    assert y.shape==(batch,1) and d['h_text'].shape==(batch,128)
    assert torch.isfinite(y).all()
    assert torch.count_nonzero(d['h_text'][0])==0
    assert torch.count_nonzero(d['text_correction'][0])==0
    assert torch.count_nonzero(d['filing_attention'][0])==0
    assert torch.count_nonzero(d['month_attention'][0])==0
    y.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_attention_normalization_padding_permutation_and_no_event_leak():
    torch.manual_seed(12)
    model=MultimodalRegressor().eval();args=inputs()
    with torch.no_grad():
        y,d=model(**args,return_diagnostics=True)
        assert d['ages'].tolist()==[5,4,3,2,1,0]
        torch.testing.assert_close(d['filing_attention'].sum(-1),args['filing_mask'].any(-1).float())
        torch.testing.assert_close(d['month_attention'].sum(-1),args['filing_mask'].any(-1).any(-1).float())
        assert torch.count_nonzero(d['filing_attention'][~args['filing_mask']])==0
        assert torch.count_nonzero(d['month_attention'][~d['month_mask']])==0
        torch.testing.assert_close(d['event_intensity'][:,0],torch.log1p(args['filing_counts'].sum(-1).float()))
        changed=dict(args,filings=args['filings'].masked_fill(~args['filing_mask'].unsqueeze(-1),1e20))
        torch.testing.assert_close(model(**changed),y,rtol=0,atol=0)
        permutation=torch.tensor([3,1,0,2])
        shuffled=dict(args,filings=args['filings'][:,:,permutation],filing_mask=args['filing_mask'][:,:,permutation])
        torch.testing.assert_close(model(**shuffled),y,atol=1e-6,rtol=1e-5)
        padded=dict(args,filings=torch.cat([args['filings'],torch.full((3,6,2,384),1e20)],2),
                    filing_mask=torch.cat([args['filing_mask'],torch.zeros(3,6,2,dtype=torch.bool)],2))
        torch.testing.assert_close(model(**padded),y,atol=1e-6,rtol=1e-5)
        expected=model.head(model.fusion_norm(d['h_quant'][0:1]))
        torch.testing.assert_close(y[0:1],expected,atol=1e-6,rtol=1e-5)


def test_all_trainable_branches_receive_gradients_and_padding_has_none():
    torch.manual_seed(23);model=MultimodalRegressor();args=inputs(batch=4,k=5)
    args['filings'].requires_grad_()
    model(**args).sum().backward()
    for name,parameter in model.named_parameters():
        assert parameter.grad is not None,name
        assert torch.isfinite(parameter.grad).all(),name
    for prefix in ('quant_encoder','event_memory.projection','event_memory.filing_key','event_memory.filing_score',
                   'event_memory.age_embedding','event_memory.temporal_key','event_memory.temporal_score',
                   'gate','text_residual','head'):
        assert any(p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith(prefix)),prefix
    assert torch.count_nonzero(args['filings'].grad[~args['filing_mask']])==0


@pytest.mark.parametrize('fault',['counts','mask','nan','months','integer','batch'])
def test_invalid_input_rejected(fault):
    args=inputs()
    if fault=='counts':args['filing_counts']+=1
    if fault=='mask':args['filing_mask']=args['filing_mask'].float()
    if fault=='nan':args['filings'][1,0,0,0]=float('nan')
    if fault=='months':args['filings']=args['filings'][:,:5]
    if fault=='integer':args['filings']=args['filings'].long()
    if fault=='batch':args['quant']=args['quant'][:1]
    with pytest.raises(ValueError):MultimodalRegressor()(**args)


def test_single_event_attention_and_empty_softmax_backward():
    scores=torch.randn(2,3,requires_grad=True);mask=torch.tensor([[False,False,False],[False,True,False]])
    weights=masked_softmax(scores,mask)
    torch.testing.assert_close(weights,torch.tensor([[0.,0.,0.],[0.,1.,0.]]))
    weights.sum().backward();assert torch.isfinite(scores.grad).all()


def test_configuration_metadata():
    config=read_multimodal_config('configs/multimodal_model.yaml')
    assert config==MultimodalConfig()
    assert model_metadata(config)['config']['quant']['d_model']==128
    with pytest.raises(ValueError):replace(config,gate_mode='invalid')


def test_actual_model_integrates_with_trainer_and_strict_reload(tmp_path):
    from scripts.verify_multimodal_model import build_smoke_components
    from src.training.multimodal import fit_multimodal,load_multimodal_checkpoint,collate_multimodal,forward_multimodal
    from src.training.trainer import TrainingConfig
    components=build_smoke_components()
    components['train_dataset']=components['train_dataset'][:8]
    components['validation_dataset']=components['validation_dataset'][:4]
    model,summary=fit_multimodal(**components,config=TrainingConfig(batch_size=4,max_epochs=2),
                                target_year=2021,output_dir=tmp_path/'run')
    restored=components['model_factory']()
    saved=load_multimodal_checkpoint(tmp_path/'run/best.pt',restored,
        expected_text_metadata=components['text_metadata'],expected_feature_names=components['feature_names'],
        expected_model_metadata=components['model_metadata'])
    assert saved['epoch']==summary['best_epoch']
    assert not any(n.startswith('text_encoder') for n in saved['trainable_parameter_names'])
    batch=collate_multimodal(components['validation_dataset'])
    with torch.no_grad():
        torch.testing.assert_close(forward_multimodal(model,batch,'cpu'),forward_multimodal(restored,batch,'cpu'),rtol=0,atol=0)

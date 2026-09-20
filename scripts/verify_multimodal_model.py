"""Verify the actual multimodal architecture; no full-dataset training."""
import argparse
from dataclasses import replace
from functools import partial
from pathlib import Path
import torch
import pyarrow.compute as pc
from scripts.run_multimodal import smoke_components, compare_reloaded_models
from src.models.multimodal_model import MultimodalRegressor, read_multimodal_config, model_metadata
from src.training.multimodal import fit_multimodal, load_multimodal_checkpoint, collate_multimodal
from src.training.trainer import TrainingConfig, _json_write, seed_everything


def build_smoke_components(year=2021, model_config='configs/multimodal_model.yaml'):
    if year != 2021:
        raise ValueError('synthetic fixture dates use the 2021 split only')
    components=smoke_components()
    config=read_multimodal_config(model_config)
    components['model_factory']=partial(MultimodalRegressor,config)
    components['model_metadata']=model_metadata(config)
    return components


def verify_real_vectors(model, cache_dir):
    from src.data.precompute_text_embeddings import load_verified_cache
    table,metadata,coverage=load_verified_cache(cache_dir,allow_partial=True)
    successful=table.filter(pc.equal(table['status'],'success')).slice(0,2).to_pylist()
    if len(successful)<2:raise ValueError('need at least two successfully cached filings')
    gen=torch.Generator().manual_seed(300)
    quant=torch.randn(3,12,147,generator=gen)
    filings=torch.zeros(3,6,1,384);mask=torch.zeros(3,6,1,dtype=torch.bool)
    filings[0,5,0]=torch.tensor(successful[0]['embedding']);mask[0,5,0]=True
    filings[1,0,0]=torch.tensor(successful[1]['embedding']);mask[1,0,0]=True
    model.eval()
    with torch.no_grad():
        prediction,details=model(quant,filings,mask,mask.sum(-1),return_diagnostics=True)
    assert torch.isfinite(prediction).all() and torch.count_nonzero(details['text_correction'][2])==0
    return {'passed':True,'scope':'Real cached embeddings with synthetic quant and artificial month placement; interface check only, not historical inference.',
            'cache_manifest_sha256':metadata['cache_manifest_sha256'],'cache_complete':coverage['complete'],
            'prediction_shape':list(prediction.shape),'predictions':prediction[:,0].tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,help='New directory')
    parser.add_argument('--model-config',default='configs/multimodal_model.yaml')
    parser.add_argument('--cache-dir',help='Optional existing real filing cache, checked explicitly in partial mode')
    args=parser.parse_args()
    torch.set_num_threads(2)
    components=build_smoke_components(model_config=args.model_config)
    output=Path(args.output_dir)
    training=TrainingConfig(batch_size=16,max_epochs=12,early_stopping_patience=5)
    model,summary=fit_multimodal(**components,config=training,target_year=2021,output_dir=output)
    restored=components['model_factory']()
    load_multimodal_checkpoint(output/'best.pt',restored,expected_text_metadata=components['text_metadata'],
        expected_feature_names=components['feature_names'],expected_model_metadata=components['model_metadata'])
    batch=collate_multimodal(components['validation_dataset'][:8])
    actual,expected=compare_reloaded_models(model,restored,batch)
    # Fixed-data learning probe uses the actual architecture, disabling dropout only for this diagnostic.
    seed_everything(42)
    config=read_multimodal_config(args.model_config)
    probe=MultimodalRegressor(replace(config,quant=replace(config.quant,dropout=0)))
    tiny=collate_multimodal(components['train_dataset'][:8])
    inputs={k:tiny[k] for k in ('quant','filings','filing_mask','filing_counts')}
    optimizer=torch.optim.AdamW(probe.parameters(),lr=.001)
    criterion=torch.nn.HuberLoss()
    losses=[]
    for step in range(50):
        optimizer.zero_grad(set_to_none=True)
        loss=criterion(probe(**inputs)[:,0],tiny['target'])
        losses.append(float(loss.detach()))
        loss.backward();optimizer.step()
    with torch.no_grad():final=float(criterion(probe(**inputs)[:,0],tiny['target']))
    assert final < losses[0]*.1, f'tiny-batch learning failed: {losses[0]} -> {final}'
    result={'passed':True,'architecture':'ModernTCN + additive filing attention + temporal attention + gated residual',
            'training_data':'synthetic','parameter_count':sum(p.numel() for p in model.parameters()),
            'summary':summary,'reload_max_absolute_difference':float((actual-expected).abs().max()),
            'tiny_batch':{'samples':8,'steps':50,'dropout':0,'initial_huber_loss':losses[0],'final_huber_loss':final},
            'input_shapes':{k:list(batch[k].shape) for k in ('quant','filings','filing_mask','filing_counts')},
            'prediction_shape':list(actual.shape)}
    if args.cache_dir:result['real_embedding_interface']=verify_real_vectors(restored,args.cache_dir)
    _json_write(output/'verification.json',result)
    print(result)

if __name__=='__main__':main()

"""Bounded real-data integration and train-only tiny-batch overfit, not OOS research."""
import argparse
from dataclasses import replace
from functools import partial
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Subset
from src.training.week2_components import build_components
from src.training.trainer import fit, TrainingConfig, seed_everything, load_checkpoint
from src.models.quant_model import QuantRegressor
from src.data.quant_dataset import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='outputs/member_c_week2')
    parser.add_argument('--dataset-config',default='configs/datasets.yaml')
    parser.add_argument('--model-config',default='configs/model.yaml')
    args=parser.parse_args()
    torch.set_num_threads(2)
    root=Path(args.output_dir)
    root.mkdir(parents=True,exist_ok=False)
    components=build_components(args.dataset_config,args.model_config,2021)
    generator=torch.Generator().manual_seed(42)
    train_ids=torch.randperm(len(components['train_dataset']),generator=generator)[:128].tolist()
    validation_ids=torch.randperm(len(components['validation_dataset']),generator=generator)[:64].tolist()
    full_counts={name:len(components[name+'_dataset']) for name in ('train','validation')}
    for name, ids in [('train',train_ids),('validation',validation_ids)]:
        components[name+'_dataset']=Subset(components[name+'_dataset'],ids)
    components['preprocessing_metadata']={**components['preprocessing_metadata'],
        'integration_check':True,'sampling_seed':42,'train_subset_indices':train_ids,
        'validation_subset_indices':validation_ids}
    model,summary=fit(**components,config=TrainingConfig(max_epochs=3,batch_size=32),
                      target_year=2021,output_dir=root/'integration')
    restored=components['model_factory']()
    load_checkpoint(root/'integration/best.pt',restored,
                    expected_feature_names=components['feature_names'],
                    expected_model_metadata=components['model_metadata'])
    sample=torch.stack([components['train_dataset'][i]['quant'] for i in range(16)])
    with torch.no_grad():
        torch.testing.assert_close(restored(sample),model(sample),rtol=0,atol=0)
    # Debug train-only memorization; dropout disabled explicitly, not selected by OOS.
    seed_everything(42)
    overfit_config=replace(model.config,dropout=0)
    debug_model=QuantRegressor(overfit_config)
    labels=torch.stack([components['train_dataset'][i]['target'] for i in range(16)]).reshape(-1,1)
    criterion=nn.HuberLoss()
    optimizer=torch.optim.AdamW(debug_model.parameters(),lr=.001,weight_decay=.0001)
    initial=criterion(debug_model(sample),labels).item()
    history=[]
    for step in range(1,201):
        optimizer.zero_grad()
        loss=criterion(debug_model(sample),labels)
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in debug_model.parameters()):
            raise ValueError('invalid gradient')
        optimizer.step()
        if step%10==0:
            with torch.no_grad(): current=criterion(debug_model(sample),labels).item()
            history.append({'step':step,'loss':current})
            print({'overfit_step':step,'loss':current},flush=True)
            if current < initial*.05: break
    if current >= initial*.1:
        raise ValueError('tiny-batch overfit did not reduce loss sufficiently')
    report={'scope':'real-data integration only; no test labels loaded; not official results',
            'full_available_counts':full_counts,'integration_train_samples':128,'integration_validation_samples':64,
            'model_parameters':sum(p.numel() for p in model.parameters()),'config':components['model_metadata'],
            'input_shape':list(sample.shape),'encoder_shape':list(model.encoder(sample).shape),
            'output_shape':list(model(sample).shape),'integration':summary,'reload_equal':True,
            'overfit':{'train_samples':16,'dropout':0,'initial_loss':initial,'final_loss':current,'history':history}}
    write_json(root/'verification.json',report)
    print(report,flush=True)


if __name__=='__main__': main()

"""Label-free monthly forecasts from verified event windows and annual weights."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.multimodal_dataset import EventStore,MultimodalDataset
from src.data.quant_dataset import sha256,write_json
from src.inference.predict_month import canonical_month,validate_predictions,CORE_COLUMNS
from src.models.modern_tcn import QuantModelConfig
from src.models.multimodal_model import MultimodalConfig,MultimodalRegressor,model_metadata
from src.training.trainer import annual_bounds
from src.training.multimodal import load_multimodal_checkpoint,collate_multimodal,forward_multimodal


def predict_month(store_dir,cache_dir,checkpoint_path,*,year,month,output_dir,batch_size=128):
    if type(batch_size) is not int or batch_size<1:raise ValueError('batch_size must be positive')
    period=canonical_month(month)
    if period.year!=year:raise ValueError('month must belong to model test year')
    output=Path(output_dir)
    if output.exists():raise FileExistsError(output)
    checkpoint_path=Path(checkpoint_path)
    checkpoint_hash=sha256(checkpoint_path)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    bounds=annual_bounds(year)
    if checkpoint['target_year']!=year or checkpoint['annual_bounds']!=bounds:
        raise ValueError('checkpoint annual split mismatch')
    for role in ('train','validation'):
        low,high=checkpoint['data_audit'][role]['target_month_range']
        if not bounds[role][0]<=low<=high<=bounds[role][1]:raise ValueError('checkpoint dates outside permitted split')
    summary=json.loads(checkpoint_path.with_name('summary.json').read_text())
    if (summary['best_epoch']!=checkpoint['epoch'] or summary['best_validation_loss']!=checkpoint['validation_loss']
        or checkpoint.get('selection_metric')!='validation_huber_loss'):
        raise ValueError('checkpoint is not completed validation selection')
    events=EventStore(cache_dir)
    if events.metadata['synthetic']:raise ValueError('production inference requires real cache provenance')
    dataset=MultimodalDataset(store_dir,events,year=year,partition='test',supervised=False,month=month)
    if checkpoint['preprocessing_metadata']!=dataset.metadata:raise ValueError('quant preprocessing/data provenance mismatch')
    raw=checkpoint['model_metadata']['config']
    config=MultimodalConfig(quant=QuantModelConfig(**raw['quant']),gate_mode=raw['gate_mode'])
    model=MultimodalRegressor(config)
    load_multimodal_checkpoint(checkpoint_path,model,expected_text_metadata=events.metadata,
                              expected_feature_names=dataset.feature_names,expected_model_metadata=model_metadata(config))
    if sha256(checkpoint_path)!=checkpoint_hash:raise ValueError('checkpoint changed while loading')
    values=[]
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_multimodal,num_workers=0):
            if 'target' in batch:raise ValueError('realized labels exposed to inference')
            prediction=forward_multimodal(model,batch,'cpu')
            if prediction.shape!=(len(batch['quant']),1):raise ValueError('invalid prediction shape')
            values.extend(prediction[:,0].tolist())
    table=pd.DataFrame(dict(target_month=month,permno=dataset.samples.permno.to_numpy(),
        predicted_excess_return=values,quant_end_month=str(period-1),model_year=year,checkpoint_sha256=checkpoint_hash))
    table=table.sort_values(['predicted_excess_return','permno'],ascending=[False,True]).reset_index(drop=True)
    table['rank']=np.arange(1,len(table)+1);table=table[CORE_COLUMNS]
    report=validate_predictions(table,month=month,expected_permnos=dataset.samples.permno,
                               model_year=year,checkpoint_sha256=checkpoint_hash)
    # A manifest written last is the completion signal. Never overwrite another run.
    output.mkdir(parents=True,exist_ok=False)
    table.to_csv(output/'predictions.csv',index=False)
    table.to_parquet(output/'predictions.parquet',index=False)
    dataset.coverage.to_parquet(output/'coverage.parquet',index=False)
    report.update(labels_accessed=False,text_metadata=events.metadata,return_unit='decimal_excess_return',
                  files_sha256={name:sha256(output/name) for name in ('predictions.csv','predictions.parquet','coverage.parquet')})
    write_json(output/'manifest.json',report)
    return report

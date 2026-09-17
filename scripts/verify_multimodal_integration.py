"""Read-only real alignment check; does not encode or fit a model."""
import argparse
from pathlib import Path
import pandas as pd
import torch
import yaml
from src.data.multimodal_dataset import EventStore,MultimodalDataset
from src.data.quant_dataset import QuantDataset,write_json
from src.training.multimodal_evaluation import event_coverage,validate_selected_events,KEYS
from src.training.multimodal import collate_multimodal
from src.models.multimodal_model import MultimodalRegressor,read_multimodal_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/multimodal_data.yaml')
    p.add_argument('--year',type=int,default=2021)
    p.add_argument('--output-dir',required=True)
    args=p.parse_args();raw=yaml.safe_load(Path(args.config).read_text())
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    events=EventStore(raw['cache_dir'])
    quant=QuantDataset(raw['store_dir'],year=args.year,partition='train',supervised=False)
    coverage=event_coverage(quant.samples[KEYS],events.events)
    joined=quant.samples[KEYS].reset_index(names='sample_index').merge(coverage,on=KEYS,validate='one_to_one')
    # Deliberately select complete windows ONLY for an interface check, not training/evaluation.
    selected=joined[joined.text_complete&joined.has_recent_filing].head(4)
    empty=joined[joined.text_complete&~joined.has_recent_filing].head(1)
    rows=[];evidence=[]
    for record in pd.concat([selected,empty]).to_dict('records'):
        row=quant[record['sample_index']]
        source=events.evidence(row['permno'],row['target_month'])
        validate_selected_events(source,permno=row['permno'],target_month=row['target_month'])
        evidence.append(dict(permno=row['permno'],target_month=row['target_month'],
                             source_indices=source.source_index.tolist(),events=len(source)))
        row.update(events.tensors(row['permno'],row['target_month']));rows.append(row)
    if not rows:raise ValueError('no complete windows for interface verification')
    batch=collate_multimodal(rows)
    torch.manual_seed(42)
    model=MultimodalRegressor(read_multimodal_config(raw['model_config'])).eval()
    inputs={key:batch[key] for key in ('quant','filings','filing_mask','filing_counts')}
    with torch.inference_mode():prediction,details=model(**inputs,return_diagnostics=True)
    assert torch.isfinite(prediction).all()
    assert torch.count_nonzero(details['text_correction'][~details['has_events']])==0
    blocked=None
    try:MultimodalDataset(raw['store_dir'],events,year=args.year,partition='train',supervised=False)
    except ValueError as exc:
        if 'incomplete text coverage' not in str(exc):raise
        blocked=str(exc)
    write_json(out/'verification.json',dict(scope='Real aligned inputs, randomly initialized model: interface check only; no performance estimate.',
        labels_accessed=False,training_run=False,encoding_run=False,
        samples=evidence,quant_shape=list(batch['quant'].shape),filings_shape=list(batch['filings'].shape),
        predictions_finite=True,empty_text_correction_zero=True,cache_counts=events.report['counts'],
        production_dataset_blocked=blocked,text_metadata=events.metadata))
    print(f'Verified {len(rows)} real aligned windows; production readiness: {blocked or "ready"}')

if __name__=='__main__':main()

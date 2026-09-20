"""Independently check saved monthly prediction files without loading labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.data.quant_dataset import QuantDataset,sha256
from src.data.splits import annual_split
from src.inference.predict_month import validate_predictions,canonical_month


def check_export(directory,store_dir,checkpoint,require_full_year=False):
    root=Path(directory)
    manifest=json.loads((root/'manifest.json').read_text())
    provenance=json.loads((root/'provenance.json').read_text())
    if manifest['status']!='complete' or manifest['checkpoint_sha256']!=sha256(checkpoint):
        raise ValueError('incomplete export or checkpoint hash mismatch')
    if provenance['checkpoint_sha256']!=manifest['checkpoint_sha256'] or provenance['year']!=manifest['year']:
        raise ValueError('provenance inconsistency')
    ds=QuantDataset(store_dir,year=manifest['year'],partition='test',supervised=False)
    if provenance['preprocessing_metadata']!=ds.metadata:
        raise ValueError('dataset metadata mismatch')
    required_inputs = {'metadata.json','quant.npy','windows.parquet','raw_context.parquet'}
    if set(provenance['input_artifacts_sha256']) != required_inputs:
        raise ValueError('input artifact inventory incomplete')
    for name,digest in provenance['input_artifacts_sha256'].items():
        if name not in ['metadata.json','quant.npy','windows.parquet','raw_context.parquet'] or sha256(Path(store_dir)/name)!=digest:
            raise ValueError('input artifact hash mismatch')
    months=provenance['requested_months']
    if not months or len(months)!=len(set(months)) or len(months)!=manifest['months']:
        raise ValueError('month coverage mismatch')
    split = annual_split(manifest['year'])
    low, high = split.test_start.strftime('%Y-%m'), split.test_end.strftime('%Y-%m')
    for month in months:
        canonical_month(month)
        if not low <= month <= high:
            raise ValueError('month outside checkpoint annual interval')
    if require_full_year:
        expected=[str(p) for p in pd.period_range(low,high,freq='M')]
        if months!=expected: raise ValueError('incomplete annual coverage')
    expected_files={f'{month}.{extension}' for month in months for extension in ['csv','parquet']}
    if set(manifest['files_sha256'])!=expected_files:
        raise ValueError('file inventory mismatch')
    reports=[]
    for month in months:
        expected_permnos=ds.samples.loc[ds.samples.target_month.eq(month),'permno'].tolist()
        tables=[]
        for extension in ['csv','parquet']:
            name=f'{month}.{extension}';path=root/name
            if sha256(path)!=manifest['files_sha256'][name]: raise ValueError('prediction file hash mismatch')
            table=pd.read_csv(path) if extension=='csv' else pd.read_parquet(path)
            report=validate_predictions(table,month=month,expected_permnos=expected_permnos,
                         model_year=manifest['year'],checkpoint_sha256=manifest['checkpoint_sha256'])
            tables.append(table)
        if not np.array_equal(tables[0].permno,tables[1].permno) or not np.allclose(tables[0].predicted_excess_return,tables[1].predicted_excess_return,rtol=0,atol=1e-15):
            raise ValueError('CSV/Parquet predictions differ')
        reports.append(report)
    if sum(row['rows'] for row in reports)!=manifest['predictions']:
        raise ValueError('total row count mismatch')
    return {'passed':True,'months':len(months),'predictions':manifest['predictions'],
            'checkpoint_sha256':manifest['checkpoint_sha256'],'labels_accessed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/inference.yaml')
    parser.add_argument('--directory',help='Override saved prediction directory')
    parser.add_argument('--require-full-year',action='store_true')
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    data=yaml.safe_load(Path(config['dataset_config']).read_text())
    print(check_export(args.directory or config['output_dir'],data['store_dir'],config['checkpoint'],args.require_full_year))


if __name__=='__main__': main()

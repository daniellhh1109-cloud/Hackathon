"""Portable project entrypoints for Windows, WSL and macOS. Run from repository root."""
import argparse
import json
import os
from pathlib import Path
import sys
import yaml


def save_yaml(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    path.write_text(yaml.safe_dump(value,sort_keys=False,allow_unicode=True),encoding='utf-8')


def setup(args):
    root=Path(args.config).parent
    raw=Path(args.data_dir).as_posix()
    project={'raw_path':f'{raw}/chars_final_with_names.parquet',
        'text_path':f'{raw}/8k_20150101_20260831_identified.parquet',
        'factor_path':'configs/factor_char_list.csv','store_dir':'data/processed/quant_v1',
        'cache_dir':'data/embeddings/minilm_v1','model_cache_dir':'data/embeddings/models',
        'dataset_config':(root/'datasets.yaml').as_posix(),'training_config':(root/'trainer.yaml').as_posix(),
        'multimodal_training_config':(root/'multimodal_trainer.yaml').as_posix(),
        'text_config':(root/'text.yaml').as_posix(),'multimodal_data_config':(root/'multimodal_data.yaml').as_posix(),
        'model_config':'configs/model.yaml','multimodal_model_config':'configs/multimodal_model.yaml',
        'training_root':'outputs/training','predictions_root':'outputs/predictions_v1',
        'tb3ms_path':f'{raw}/TB3MS.csv','sp500_path':f'{raw}/SP500.csv',
        'device':args.device,'cpu_threads':2,
        'portfolio':{'controller':{'kind':'mock'},'policy':{},
          'filters':{'min_price':5.,'min_market_cap':1e9,'min_dollar_volume':1e7},
          'costs':{'transaction_cost_bps':10,'annual_borrow_bps':0}}}
    # Fail before writing any generated config if one exists.
    targets=[Path(args.config),root/'datasets.yaml',root/'trainer.yaml',root/'multimodal_trainer.yaml',root/'text.yaml',root/'multimodal_data.yaml']
    if any(p.exists() for p in targets):raise FileExistsError('use a new config directory or explicitly edit existing settings')
    save_yaml(project['dataset_config'],{'raw_path':project['raw_path'],'factor_path':project['factor_path'],
        'store_dir':project['store_dir'],'report_dir':'outputs/data_audit','year':2021})
    for key,base in [('training_config','configs/trainer.yaml'),('multimodal_training_config','configs/multimodal_training.yaml')]:
        config=yaml.safe_load(Path(base).read_text());config['training']['device']=args.device
        if args.device=='cuda':config['training']['batch_size']=128 if key=='multimodal_training_config' else 512
        save_yaml(project[key],config)
    text=yaml.safe_load(Path('configs/text_embeddings.yaml').read_text())
    text.update(source_path=project['text_path'],cache_dir=project['cache_dir'],model_cache_dir=project['model_cache_dir'])
    text['encoder']['device']=args.device;save_yaml(project['text_config'],text)
    save_yaml(project['multimodal_data_config'],{'store_dir':project['store_dir'],'cache_dir':project['cache_dir'],
                                              'model_config':project['multimodal_model_config']})
    save_yaml(args.config,project)
    return {'config':args.config,'device':args.device,'data_dir':raw}


def doctor():
    import importlib
    import torch
    import cvxpy
    versions={name:str(importlib.import_module(name).__version__) for name in ['torch','numpy','pandas','pyarrow','cvxpy','transformers']}
    return {'python':sys.version,'executable':sys.executable,'versions':versions,
            'cuda_available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'solvers':cvxpy.installed_solvers()}


def audit(config):
    import pyarrow.parquet as pq
    from src.data.prepare_quant import load_factors
    factors=load_factors(config['factor_path']);result={}
    for name,path in [('quant',config['raw_path']),('text',config['text_path'])]:
        f=pq.ParquetFile(path);result[name]={'path':path,'rows':f.metadata.num_rows,'columns':len(f.schema_arrow.names)}
        required=factors+['permno','date','eom','ret_exc_lead1m'] if name=='quant' else ['document_id','permno','filing_date','text']
        missing=set(required)-set(f.schema_arrow.names)
        if missing:raise ValueError(f'{name} missing columns: {sorted(missing)}')
    result['factor_count']=len(factors)
    return result


def prepare(config):
    from src.data.quant_dataset import prepare_store,QuantDataset,sha256
    from src.data.prepare_quant import load_factors
    if Path(config['store_dir']).exists():
        ds=QuantDataset(config['store_dir'],year=2021,partition='test',supervised=False)
        if ds.metadata['source']['sha256'] != sha256(config['raw_path']):
            raise ValueError('cached quant source differs from raw_path; choose a new store_dir')
        if ds.feature_names != load_factors(config['factor_path']):
            raise ValueError('cached factor order differs from factor_path; choose a new store_dir')
        return {'reused_verified_store':True,'test_samples':len(ds)}
    prepare_store(config['raw_path'],config['factor_path'],config['store_dir'])
    return {'prepared':config['store_dir']}


def embed(config,args):
    from src.data.precompute_text_embeddings import read_config,run_pipeline
    raw,encoder=read_config(config['text_config'])
    report=run_pipeline(raw['source_path'],raw['cache_dir'],encoder,model_cache_dir=raw['model_cache_dir'],
        max_documents=args.max_documents,audit_only=False,retry_failed=args.retry_failed,allow_download=args.allow_download)
    return {k:v for k,v in report.items() if k!='by_filing_month'}


def train(config,kind,year):
    import torch
    from src.training.trainer import read_config,fit,load_checkpoint
    torch.set_num_threads(config['cpu_threads'])
    output=Path(config['training_root'])/kind/str(year)
    if kind=='quant':
        from src.training.quant_components import build_components
        c=build_components(config['dataset_config'],config['model_config'],year)
        model,summary=fit(**c,config=read_config(config['training_config']),target_year=year,output_dir=output)
        restored=c['model_factory']();load_checkpoint(output/'best.pt',restored,
            expected_feature_names=c['feature_names'],expected_model_metadata=c['model_metadata'])
    else:
        from src.training.multimodal_components import build_components
        from src.training.multimodal import fit_multimodal,load_multimodal_checkpoint
        c=build_components(year=year,config_path=config['multimodal_data_config'])
        model,summary=fit_multimodal(**c,config=read_config(config['multimodal_training_config']),target_year=year,output_dir=output)
        restored=c['model_factory']();load_multimodal_checkpoint(output/'best.pt',restored,
          expected_text_metadata=c['text_metadata'],expected_feature_names=c['feature_names'],expected_model_metadata=c['model_metadata'])
    return {'checkpoint':str(output/'best.pt'),'summary':summary}


def predict(config,kind,year):
    import torch
    torch.set_num_threads(config['cpu_threads'])
    checkpoint=Path(config['training_root'])/kind/str(year)/'best.pt'
    root=Path(config['predictions_root'])/kind/str(year)
    if kind=='quant':
        from src.inference.predict_month import MonthlyPredictor
        return MonthlyPredictor(config['store_dir'],checkpoint,year=year).export(root)
    from src.inference.predict_multimodal import predict_month
    return [predict_month(config['store_dir'],config['cache_dir'],checkpoint,year=year,month=f'{year}-{m:02d}',
                         output_dir=root/f'{year}-{m:02d}',batch_size=64) for m in range(1,9 if year==2026 else 13)]


def backtest(config,args):
    from src.portfolio.workflow import run_backtest
    c=dict(config['portfolio']);c.update(raw_path=config['raw_path'],tb3ms_path=config['tb3ms_path'],sp500_path=config['sp500_path'],
        predictions_dir=args.predictions_dir or str(Path(config['predictions_root'])/args.kind),
        start_month=f'{args.start}-01-01',end_month=f'{args.end}-'+('08-01' if args.end==2026 else '12-01'))
    return run_backtest(c,args.output_dir)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['setup','doctor','audit','prepare','embed','train','predict','backtest','report','all'])
    p.add_argument('--config',default='configs/local/project.yaml')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--data-dir',default='data/raw')
    p.add_argument('--kind',choices=['quant','multimodal'],default='quant')
    p.add_argument('--year',type=int,choices=range(2021,2027),default=2021)
    p.add_argument('--start',type=int,choices=range(2021,2027),default=2021)
    p.add_argument('--end',type=int,choices=range(2021,2027),default=2026)
    p.add_argument('--output-dir');p.add_argument('--predictions-dir')
    p.add_argument('--run-dir');p.add_argument('--pptx',action='store_true')
    p.add_argument('--max-documents',type=int);p.add_argument('--allow-download',action='store_true');p.add_argument('--retry-failed',action='store_true')
    args=p.parse_args(argv)
    if args.start>args.end:p.error('start must not exceed end')
    if args.action in ('backtest','all') and not args.output_dir:p.error('--output-dir is required')
    if args.action=='report' and not args.run_dir:p.error('--run-dir is required')
    if args.max_documents is not None and args.max_documents<1:p.error('--max-documents must be positive')
    if args.action=='all' and args.max_documents is not None:p.error('all requires a complete embedding cache; no --max-documents')
    if args.action=='setup':result=setup(args)
    elif args.action=='doctor':result=doctor()
    elif args.action=='report':
        from src.portfolio.reporting import build_report,export_pptx
        result=build_report(args.run_dir)
        if args.pptx:result['pptx']=str(export_pptx(args.run_dir))
    else:
        config=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
        if args.action=='audit':result=audit(config)
        elif args.action=='prepare':result=prepare(config)
        elif args.action=='embed':result=embed(config,args)
        elif args.action=='train':result=train(config,args.kind,args.year)
        elif args.action=='predict':result=predict(config,args.kind,args.year)
        elif args.action=='backtest':result=backtest(config,args)
        else:
            audit(config);prepare(config)
            if args.kind=='multimodal':embed(config,args)
            for year in range(args.start,args.end+1):
                train(config,args.kind,year);predict(config,args.kind,year)
            result=backtest(config,args)
            from src.portfolio.reporting import build_report
            build_report(args.output_dir)
    print(json.dumps(result,indent=2,ensure_ascii=False,default=str,allow_nan=False))
    return 0

if __name__=='__main__':raise SystemExit(main())

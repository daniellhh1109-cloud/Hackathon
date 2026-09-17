"""Audit real cache readiness, verify paired evaluation, or compare frozen predictions."""
import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.precompute_text_embeddings import load_verified_cache,sha256_file,json_write
from src.training.multimodal_evaluation import event_coverage,compare_predictions,diagnostic_summary,KEYS,validate_keys
from src.training.trainer import TrainingConfig,fit,annual_bounds
from src.training.multimodal import fit_multimodal,collate_multimodal
from src.models.quant_model import QuantRegressor,model_metadata as quant_metadata
from scripts.verify_multimodal_model import build_smoke_components


def read_table(path):
    return pd.read_parquet(path) if Path(path).suffix=='.parquet' else pd.read_csv(path,dtype={'target_month':str})


def write_comparison(output,table,summaries,monthly,context):
    output=Path(output)
    table.to_parquet(output/'scored_rows.parquet',index=False)
    pd.DataFrame(summaries).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(monthly).to_csv(output/'monthly_metrics.csv',index=False)
    json_write(output/'evaluation.json',{'context':context,'metrics':summaries})
    lines=['# 多模态与数值模型对比','',context['scope'],'',
           '两种模型必须覆盖完全相同的证券—目标月份；共同使用有限的真实标签评分。文本存在分组由事件覆盖表定义，不按收益筛选。','',
           '| 分组 | 模型 | 预测数 | 评分数 | 零预测基准 R² | Huber loss | 月均 rank IC |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    def fmt(value):return '未定义' if value is None else f'{value:.6g}'
    for r in summaries:
        lines.append(f"| {r['group']} | {r['model']} | {r['n_predictions']} | {r['n_scored']} | {fmt(r.get('oos_r2_zero'))} | {fmt(r.get('huber_loss'))} | {fmt(r.get('mean_monthly_rank_ic'))} |")
    lines+=['','R² = 1 - Σ(y - prediction)² / Σy²；基准是零超额收益预测。仅在实际样本外数据上才能解释为 OOS 结果。',
            '收益单位为小数，0.01 表示 1%。有/无事件的分组不表示因果效应。',
            '2021 年测试结果此前已查看；不得据此反复调参后继续宣称测试集未触碰。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def audit(args):
    from src.data.quant_dataset import QuantDataset
    data=yaml.safe_load(Path(args.dataset_config).read_text())
    text=yaml.safe_load(Path(args.cache_config).read_text())
    table,metadata,report=load_verified_cache(text['cache_dir'],allow_partial=True)
    events=table.select(['source_index','permno','filing_date','available_at_utc','status']).to_pandas()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=False)
    summary={}
    for partition in ('train','validation','test'):
        ds=QuantDataset(data['store_dir'],year=args.year,partition=partition,supervised=False)
        coverage=event_coverage(ds.samples[KEYS],events)
        coverage.to_parquet(output/f'{partition}_coverage.parquet',index=False)
        summary[partition]={'samples':len(coverage),'with_events':int(coverage.has_recent_filing.sum()),
            'without_events':int((~coverage.has_recent_filing).sum()),
            'complete_event_windows':int(coverage.text_complete.sum()),
            'incomplete_event_windows':int((~coverage.text_complete).sum())}
        print(partition,summary[partition],flush=True)
    result={'year':args.year,'cache_counts':report['counts'],'cache_manifest_sha256':metadata['cache_manifest_sha256'],
            'partitions':summary,'realized_labels_read':False,'scope':'Readiness audit only; no predictions or performance estimates.',
            'full_universe_text_ready':all(r['incomplete_event_windows']==0 for r in summary.values()),
            'comparison_status':'not_run_requires_real_aligned_dataset_and_trained_multimodal_checkpoint'}
    json_write(output/'readiness.json',result)
    (output/'report.md').write_text('# 真实数据评估准备情况\n\n'+
        '这里只审计数据可用性，不读取收益标签，不按缓存是否成功筛选投资股票池。\n\n'+
        '\n'.join(f"- {p}: {r['samples']} 个样本，{r['with_events']} 个存在可用时间范围内公告，{r['incomplete_event_windows']} 个事件窗口编码不完整。" for p,r in summary.items())+
        '\n\n完整事件数据集、训练完成的正式多模态模型和同股票池预测尚未提供，因此不能判断文本是否提高真实收益预测。\n',encoding='utf-8')


def smoke(args):
    torch.set_num_threads(2)
    components=build_smoke_components()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=False)
    training=TrainingConfig(batch_size=16,max_epochs=12,early_stopping_patience=5)
    quant_config=components['model_factory'].args[0].quant
    qcomponents={k:v for k,v in components.items() if k!='text_metadata'}
    qcomponents.update(model_factory=partial(QuantRegressor,quant_config),model_metadata=quant_metadata(quant_config))
    for key in ('train_dataset','validation_dataset'):
        qcomponents[key]=[{k:v for k,v in row.items() if k not in ('filings','filing_mask','filing_counts')} for row in components[key]]
    quant,qsummary=fit(**qcomponents,config=training,target_year=2021,output_dir=output/'quant_training')
    multimodal,msummary=fit_multimodal(**components,config=training,target_year=2021,output_dir=output/'multimodal_training')
    # Save forecasts and their hashes BEFORE assembling the held-out validation labels.
    rows=components['validation_dataset']
    batch=collate_multimodal(rows)
    inputs={k:batch[k] for k in ('quant','filings','filing_mask','filing_counts')}
    with torch.no_grad():
        qpred=quant(batch['quant'])[:,0].numpy()
        mpred,details=multimodal(**inputs,return_diagnostics=True)
        mpred=mpred[:,0].numpy()
    keys=pd.DataFrame([{k:row[k] for k in KEYS} for row in rows])
    qtable=keys.assign(predicted_excess_return=qpred);mtable=keys.assign(predicted_excess_return=mpred)
    qtable.to_parquet(output/'quant_predictions.parquet',index=False)
    mtable.to_parquet(output/'multimodal_predictions.parquet',index=False)
    lock={name:sha256_file(output/name) for name in ('quant_predictions.parquet','multimodal_predictions.parquet')}
    lock.update(synthetic=True,partition='validation',year=2021,
                quant_checkpoint_sha256=sha256_file(output/'quant_training/best.pt'),
                multimodal_checkpoint_sha256=sha256_file(output/'multimodal_training/best.pt'))
    json_write(output/'prediction_lock.json',lock)
    labels=keys.assign(realized_target=[float(row['target']) for row in rows])
    coverage=keys.assign(has_recent_filing=[bool(row['filing_mask'].any()) for row in rows],text_complete=True)
    table,summaries,monthly=compare_predictions(qtable,mtable,labels,coverage,huber_delta=training.huber_delta)
    labels.to_parquet(output/'labels.parquet',index=False)
    coverage.to_parquet(output/'coverage.parquet',index=False)
    json_write(output/'diagnostics.json',diagnostic_summary(details))
    context={'synthetic':True,'partition':'validation','huber_delta':training.huber_delta,
             'scope':'人工数据、实际模型架构、共同验证样本的工具验收。该验证集也用于 early stopping，不是未触碰测试集，更不是比赛成绩。',
             'quant_summary':qsummary,'multimodal_summary':msummary}
    write_comparison(output,table,summaries,monthly,context)


def checkpoint_contract(quant,multimodal, *, year, partition, text_metadata):
    bounds=annual_bounds(year)
    for checkpoint in (quant,multimodal):
        if checkpoint.get('target_year')!=year or checkpoint.get('annual_bounds')!=bounds:
            raise ValueError('checkpoint annual training split mismatch')
        if checkpoint.get('selection_metric')!='validation_huber_loss':raise ValueError('unsupported checkpoint selection policy')
        for role in ('train','validation'):
            low,high=checkpoint['data_audit'][role]['target_month_range']
            if not bounds[role][0]<=low<=high<=bounds[role][1]:raise ValueError('checkpoint samples outside permitted training/validation dates')
    if quant['feature_names']!=multimodal['feature_names'] or quant['preprocessing_metadata']!=multimodal['preprocessing_metadata']:
        raise ValueError('checkpoints have different factors/preprocessing')
    if quant['training']['huber_delta']!=multimodal['training']['huber_delta']:
        raise ValueError('checkpoint loss scales differ')
    if quant['model_metadata']['name']!='QuantRegressor' or multimodal['model_metadata']['name']!='MultimodalRegressor':
        raise ValueError('expected actual quant and multimodal architectures')
    if multimodal['integration_metadata']['text']!=text_metadata:
        raise ValueError('checkpoint/cache text provenance mismatch')
    if text_metadata['synthetic']:raise ValueError('real comparison cannot use synthetic text provenance')
    return bounds[partition]


def compare(args):
    qpath,mpath=Path(args.quant_predictions),Path(args.multimodal_predictions)
    artifacts={'quant_predictions':qpath,'multimodal_predictions':mpath,
               'quant_checkpoint':Path(args.quant_checkpoint),'multimodal_checkpoint':Path(args.multimodal_checkpoint),
               'universe':Path(args.universe)}
    hashes={name+'_sha256':sha256_file(path) for name,path in artifacts.items()}
    qtable,mtable=read_table(qpath),read_table(mpath)
    qckpt=torch.load(args.quant_checkpoint,map_location='cpu',weights_only=True)
    mckpt=torch.load(args.multimodal_checkpoint,map_location='cpu',weights_only=True)
    cache,metadata,_=load_verified_cache(args.cache_dir,allow_partial=True)
    low,high=checkpoint_contract(qckpt,mckpt,year=args.year,partition=args.partition,text_metadata=metadata)
    keys=validate_keys(read_table(args.universe),'independent universe')[KEYS]
    if not keys.target_month.between(low,high).all():raise ValueError('evaluation universe outside requested split')
    coverage=event_coverage(keys,cache.select(['source_index','permno','filing_date','available_at_utc','status']).to_pandas())
    # Preflight both predictions on a dummy label table BEFORE loading actual labels.
    dummy=keys.assign(realized_target=np.nan)
    compare_predictions(qtable,mtable,dummy,coverage,huber_delta=qckpt['training']['huber_delta'])
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=False)
    lock={**hashes,'cache_manifest_sha256':metadata['cache_manifest_sha256'],
          'year':args.year,'partition':args.partition,'return_unit':'decimal_excess_return',
          'provenance_limit':'Input hashes record supplied artifacts; this evaluator does not independently regenerate their predictions.'}
    for name,path in artifacts.items():
        if sha256_file(path)!=hashes[name+'_sha256']:raise ValueError('input artifact changed during inspection')
    json_write(output/'prediction_lock.json',lock)
    labels=read_table(args.labels)
    # Detect on-disk changes during input inspection; evaluate only the locked artifacts.
    for path,digest in [(qpath,lock['quant_predictions_sha256']),(mpath,lock['multimodal_predictions_sha256'])]:
        if sha256_file(path)!=digest:raise ValueError('prediction file changed during evaluation')
    table,summaries,monthly=compare_predictions(qtable,mtable,labels,coverage,huber_delta=qckpt['training']['huber_delta'])
    coverage.to_parquet(output/'coverage.parquet',index=False)
    write_comparison(output,table,summaries,monthly,{'synthetic':False,**lock,
        'scope':'真实数据、已冻结预测的同股票池对比；标签仅在预测与缓存检查并记录哈希后读取。不执行训练或参数修改。'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='mode',required=True)
    p=subs.add_parser('audit');p.set_defaults(action=audit)
    p.add_argument('--dataset-config',default='configs/datasets.yaml')
    p.add_argument('--cache-config',default='configs/text_embeddings.yaml');p.add_argument('--year',type=int,default=2021)
    p=subs.add_parser('smoke');p.set_defaults(action=smoke)
    p=subs.add_parser('compare');p.set_defaults(action=compare)
    for flag in ('quant-predictions','multimodal-predictions','quant-checkpoint','multimodal-checkpoint','universe','labels','cache-dir'):
        p.add_argument('--'+flag,required=True)
    p.add_argument('--year',type=int,required=True)
    p.add_argument('--partition',choices=['train','validation','test'],required=True)
    for child in subs.choices.values():child.add_argument('--output-dir',required=True)
    args=parser.parse_args();args.action(args)

if __name__=='__main__':main()

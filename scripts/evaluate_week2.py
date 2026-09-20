"""D's baseline fitting and locked-prediction evaluation; no OOS retuning loop."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import yaml

from src.data.quant_dataset import QuantDataset, make_loader, sha256, write_json
from src.models.window_ridge import fit_ridge,predict_ridge
from src.models.modern_tcn import QuantModelConfig
from src.models.quant_model import QuantRegressor,model_metadata
from src.training.trainer import load_checkpoint,read_config
from src.training.evaluation import evaluate_table,label_scale


def baseline(config):
    root=Path(config['output_dir'])/'baseline'
    root.mkdir(parents=True,exist_ok=False)
    data=yaml.safe_load(Path(config['dataset_config']).read_text())
    training=read_config(config['trainer_config'])
    train=QuantDataset(data['store_dir'],year=config['year'],partition='train')
    val=QuantDataset(data['store_dir'],year=config['year'],partition='validation')
    reports={}
    for mode in ['latest','window']:
        print(f'Fitting {mode} ridge on {len(train)} train / {len(val)} validation rows...',flush=True)
        model,report=fit_ridge(train,val,mode=mode,alphas=config['ridge_alphas'],delta=training.huber_delta)
        np.savez(root/f'ridge_{mode}.npz',coef=model['coef'],intercept=model['intercept'],alpha=model['alpha'])
        reports[mode]=report
        print({'mode':mode,'selected_alpha':model['alpha']},flush=True)
    write_json(root/'selection.json',{'year':config['year'],'data_metadata':train.metadata,
                                    'reports':reports,'huber_delta':training.huber_delta,
                                    'config':config,'test_labels_accessed':False})
    # Scale review deliberately uses training and validation only.
    write_json(root/'label_scale.json',{part:label_scale(ds.labels[ds.rows],training.huber_delta)
               for part,ds in [('train',train),('validation',val)]})


def evaluate(config):
    root=Path(config['output_dir'])
    directory=root/'evaluation'
    if directory.exists(): raise FileExistsError(directory)
    data=yaml.safe_load(Path(config['dataset_config']).read_text())
    checkpoint=torch.load(config['checkpoint'],map_location='cpu',weights_only=True)
    if checkpoint['target_year'] != config['year']:
        raise ValueError('checkpoint year mismatch')
    selection=json.loads((root/'baseline/selection.json').read_text())
    if selection['year'] != config['year']:
        raise ValueError('baseline year mismatch')
    model_config=QuantModelConfig(**checkpoint['model_metadata']['config'])
    neural=QuantRegressor(model_config)
    load_checkpoint(config['checkpoint'],neural,expected_feature_names=selection['data_metadata']['feature_names'],
                    expected_model_metadata=model_metadata(model_config))
    if checkpoint['preprocessing_metadata'] != selection['data_metadata']:
        raise ValueError('checkpoint and baseline did not use the same dataset metadata')
    delta=checkpoint['training']['huber_delta']
    if delta != selection['huber_delta']:
        raise ValueError('selection loss delta mismatch')
    # Completion of training is required; don't evaluate an intermediate checkpoint.
    summary=json.loads(Path(config['checkpoint']).with_name('summary.json').read_text())
    if summary['best_epoch'] != checkpoint['epoch']:
        raise ValueError('training summary/checkpoint mismatch')
    models={}
    for mode in ['latest','window']:
        with np.load(root/f'baseline/ridge_{mode}.npz',allow_pickle=False) as archive:
            models['ridge_'+mode]={'mode':mode,'coef':archive['coef'],'intercept':float(archive['intercept'])}
    directory.mkdir()
    # Freeze choices and artifact identities BEFORE constructing/scoring OOS predictions.
    lock={'year':config['year'],'checkpoint_sha256':sha256(config['checkpoint']),
          'baseline_selection_sha256':sha256(root/'baseline/selection.json'),
          'baseline_model_sha256':{name:sha256(root/f'baseline/{name}.npz') for name in models},
          'selection_metric':'validation_huber_loss','test_used_for_selection':False,
          'selected_epoch':checkpoint['epoch']}
    write_json(directory/'selection_lock.json',lock)
    prediction_paths={}; datasets={}
    for part in ['train','validation','test']:
        ds=QuantDataset(data['store_dir'],year=config['year'],partition=part,supervised=False)
        if ds.metadata != selection['data_metadata']: raise ValueError('current dataset metadata changed')
        datasets[part]=ds
        table=ds.samples[['permno','target_month','row_index']].copy()
        values=[]
        with torch.no_grad():
            for batch in make_loader(ds,batch_size=512):
                values.append(neural(batch['quant']).squeeze(1).numpy())
        table['modern_tcn']=np.concatenate(values)
        for name,model in models.items(): table[name]=predict_ridge(ds,model)
        table['zero']=0.0
        path=directory/f'{part}_predictions.parquet'
        table.to_parquet(path,index=False)
        prediction_paths[part]=path
        print(f'Frozen {part}: {len(table)} predictions/model',flush=True)
    write_json(directory/'prediction_lock.json',{name:sha256(path) for name,path in prediction_paths.items()})
    # First access to test realized targets occurs only AFTER every prediction is saved.
    labels=np.load(Path(data['store_dir'])/'labels.npy',mmap_mode='r',allow_pickle=False)
    all_summaries=[]; all_monthly=[]
    for part,path in prediction_paths.items():
        table=pd.read_parquet(path)
        table['realized_target']=labels[table.row_index.to_numpy()]
        summary,monthly=evaluate_table(table,['modern_tcn','ridge_latest','ridge_window','zero'],delta)
        all_summaries.extend([{'partition':part,**row} for row in summary])
        all_monthly.extend([{'partition':part,**row} for row in monthly])
    pd.DataFrame(all_summaries).to_csv(directory/'metrics.csv',index=False)
    pd.DataFrame(all_monthly).to_csv(directory/'monthly_metrics.csv',index=False)
    write_json(directory/'metrics.json',all_summaries)
    write_report(root,all_summaries,checkpoint,selection)
    write_plots(root,all_summaries)


def write_report(root,results,checkpoint,selection):
    def percent(value): return 'undefined' if value is None else f'{value:.4%}'
    lines=['# 第二周成员 D：2021 年预测评估','',
        '所有模型使用相同证券—月份、相同 147 因子排名缓存。预测保存并记录哈希后才读取测试目标；没有按测试表现重新选择模型。',
        '',f"ModernTCN 最佳轮次：{checkpoint['epoch']}；选择准则：验证 Huber loss，delta={checkpoint['training']['huber_delta']}。",
        f"Ridge alpha：latest={selection['reports']['latest']['selected_alpha']}，window={selection['reports']['window']['selected_alpha']}。",'',
        '| 分区 | 模型 | 预测数 | 有标签数 | 零预测 R² | RMSE | 平均月度 rank IC |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in results:
        ic='undefined' if row['mean_monthly_rank_ic'] is None else f"{row['mean_monthly_rank_ic']:.5f}"
        lines.append(f"| {row['partition']} | {row['model']} | {row['n_predictions']} | {row['n_scored']} | {percent(row['oos_r2_zero'])} | {row['rmse']:.6f} | {ic} |")
    test_rows = {row['model']: row for row in results if row['partition'] == 'test'}
    if all(test_rows['modern_tcn']['oos_r2_zero'] < test_rows[name]['oos_r2_zero'] for name in ('ridge_latest', 'ridge_window')):
        lines.extend(['', '本次 ModernTCN 测试 R² 低于两条 Ridge 基线。复杂模型尚未带来预测误差上的优势；不据此回调本次测试模型。'])
    lines.extend(['','## 解释与边界','',
        '- R² = 1 − Σ(y−预测)² / Σy²，参照零超额收益预测，不是均值中心化的 sklearn R²。负数表示平方误差比零预测更大。',
        '- RMSE/MAE/Huber 的输入都是收益小数，0.01 表示 1%；不是把百分数直接喂入模型。',
        '- rank IC 每月分别做并列平均排名后计算 Pearson 相关，再等权平均有效月份；常数序列或少于 3 个有效样本时留空。',
        '- ridge_latest 只看末月 147 维；ridge_window 按月展开 12×147=1764 维，与神经网络看到相同历史。两者均仅训练期拟合截距，无额外 StandardScaler。',
        '- 这里复用第一周 Ridge 的正则回归范式，但使用 B 的平均排名和窗口合格样本；不把第一周官方 dense-rank/StandardScaler 的不同样本结果直接并列当作同口径比较。',
        '- 训练/验证预测文件包括所有窗口合格行，缺失标签只在评分中排除；有限标签的行与实际拟合/验证样本完全一致。',
        '- 验证结果参与 epoch/alpha 选择，不能视为样本外证据。仅 2021 年是本次测试期；未声称完成 2022–2026 或组合回测。',
        '- 现有代码和原始数据曾用于此前探索，因此本次是固定协议的历史样本外评估，不宣称从未接触过该历史时期。',
        '- Huber delta=1 对应 100 个百分点的预测误差；它不是 1%。线性区误差比例见 metrics.csv，标签尺度诊断仅使用训练/验证，见 baseline/label_scale.json。',
        '- 不根据本次测试输赢回调参数；后续调整仍应以训练/验证为依据，并保留此次结果。',''])
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def write_plots(root,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=pd.read_csv(root/'quant_training/history.csv')
    figure,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].plot(history.epoch,history.train_loss,label='Train (dropout on)')
    axes[0].plot(history.epoch,history.validation_loss,label='Validation (eval)')
    axes[0].set(xlabel='Epoch',ylabel='Huber loss',title='2021 model: training and validation')
    axes[0].legend()
    test=[row for row in results if row['partition']=='test']
    axes[1].bar([row['model'] for row in test],[100*row['oos_r2_zero'] for row in test])
    axes[1].axhline(0,color='black',linewidth=.8)
    axes[1].set(ylabel='R-squared vs zero forecast (%)',title='2021 test: same samples and normalization')
    axes[1].tick_params(axis='x',rotation=20)
    figure.tight_layout()
    figure.savefig(root/'comparison.png',dpi=160)
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/member_d_week2.yaml')
    parser.add_argument('--stage',choices=['baseline','evaluate','all'],default='all')
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.stage in ('baseline','all'): baseline(config)
        if args.stage in ('evaluate','all'): evaluate(config)


if __name__=='__main__': main()

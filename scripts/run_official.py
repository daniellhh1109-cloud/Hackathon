"""Run the supplied official linear algorithm adapted to the 2026 data release."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shlex
import sys
import time
import numpy as np
import pandas as pd
import sklearn
import pyarrow
from scripts.run_member_d import sha256, write_json
from src.data.prepare_quant import load_factors
from src.data.build_samples import build_baseline_panel
from src.data.splits import annual_split
from src.models.official_linear_baseline import dense_monthly_transform, fit_official_year, official_grids
from src.training.metrics import prediction_metrics


def universe_mask(panel, cfg):
    """Full mode retains each observed stock-month, including later entrants."""
    mode = cfg.get('universe', 'initial_smoke')
    if mode == 'all_available':
        if 'n_securities' in cfg or 'selection_month' in cfg:
            raise ValueError('full universe cannot specify smoke selection')
        return pd.Series(True, index=panel.index)
    if mode != 'initial_smoke':
        raise ValueError('unknown universe')
    count = cfg['n_securities']
    if cfg['selection_month'] != '2015-01-31' or type(count) is not int or count < 2:
        raise ValueError('invalid initial-universe selection')
    selected = sorted(panel.loc[panel.eom == pd.Timestamp(cfg['selection_month']).date(), 'permno'].unique())[:count]
    if len(selected) != count:
        raise ValueError('insufficient initial securities')
    return panel.permno.isin(selected)


def check_predictions(predictions, panel, years, models):
    expected = {d.date() for year in years for d in pd.date_range(
        annual_split(year).test_start, annual_split(year).test_end, freq='MS')}
    if predictions.duplicated(['permno', 'target_month']).any():
        raise ValueError('duplicate prediction key')
    if set(predictions.target_month) != expected:
        raise ValueError('missing test month')
    wanted = panel.loc[panel.target_month.isin(expected), ['permno', 'target_month']]
    keys = pd.MultiIndex.from_frame(predictions[['permno', 'target_month']])
    wanted_keys = pd.MultiIndex.from_frame(wanted)
    if len(keys) != len(wanted_keys) or len(wanted_keys.difference(keys)):
        raise ValueError('prediction universe differs from available stock-months')
    if not np.isfinite(predictions[models].to_numpy()).all():
        raise ValueError('non-finite predictions')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/member_d_official_full.json')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg['preprocessing'] != 'full_same_month_median_dense_rank_then_train_standard_scaler' or cfg['alpha_grids'] != 'official_original_full_grids' or cfg['models'] != ['ols', 'lasso', 'ridge', 'en']:
        raise ValueError('unsupported algorithm configuration')
    years = cfg['test_years']
    if not years or len(years) != len(set(years)):
        raise ValueError('invalid test years')
    for year in years: annual_split(year)
    years = sorted(years)
    if cfg.get('require_full_period', False) and years != list(range(2021, 2027)):
        raise ValueError('full experiment requires years 2021 through 2026')
    start = time.monotonic()
    factors = load_factors(cfg['factor_list'])
    if cfg.get('verified_factors') and factors != load_factors(cfg['verified_factors']):
        raise ValueError('factor list mismatch')
    input_keys = [k for k in ['chars','factor_list','verified_factors','official_prediction_script','official_portfolio_script'] if cfg.get(k)]
    input_hashes = {cfg[k]: sha256(cfg[k]) for k in input_keys}
    out = Path(cfg['output_dir'])
    if out.exists() and any(out.iterdir()):
        raise ValueError('output directory is not empty; choose a new output_dir')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'config.json', cfg)
    columns = list(dict.fromkeys(['permno','date','eom','ret_exc_lead1m','beta_60m','me','prc'] + factors))
    print('Load actual data and validate calendar keys...', flush=True)
    raw = pd.read_parquet(cfg['chars'], columns=columns)
    raw = raw[pd.to_datetime(raw.eom) < pd.Timestamp(annual_split(max(years)).test_end)]
    panel = build_baseline_panel(raw); del raw
    mask = universe_mask(panel, cfg)
    print(f'Dense ranking full monthly cross-sections: {len(panel)} rows...', flush=True)
    features, diagnostics = dense_monthly_transform(panel, factors)
    panel, features = panel.loc[mask].reset_index(drop=True), features.loc[mask].reset_index(drop=True)
    selected = sorted(panel.permno.unique())
    annual, outputs = [], []
    params_by_year = {}
    for year in years:
        print(f'{year}: fit original OLS / 81 Lasso / 91 Ridge / 81 Elastic Net candidates...', flush=True)
        prediction, report, parameters = fit_official_year(panel, features, factors, year)
        report['boundaries'] = asdict(annual_split(year))
        prediction = prediction.merge(panel[['permno','eom','beta_60m','me','prc']], on=['permno','eom'], validate='one_to_one')
        check_predictions(prediction, panel, [year], cfg['models'])
        # Keep completed annual fits even if a subsequent year fails.
        prediction.to_csv(out/f'predictions_{year}.csv', index=False)
        write_json(out/f'parameters_{year}.json', parameters)
        write_json(out/f'results_{year}.json', report)
        annual.append(report); outputs.append(prediction); params_by_year[str(year)] = parameters
    predictions = pd.concat(outputs, ignore_index=True).sort_values(['target_month','permno'])
    expected = check_predictions(predictions, panel, years, cfg['models'])
    pooled = {m: prediction_metrics(predictions.ret_exc_lead1m, predictions[m]) for m in cfg['models']}
    # Verify exported scaler + coefficients reconstruct every prediction.
    max_error = 0.0
    for year in years:
        par = params_by_year[str(year)]
        rows = panel[panel.target_month.map(lambda m: annual_split(year).partition(m) == 'test')]
        x = (features.loc[rows.index].to_numpy() - np.array(par['scaler_mean'])) / np.array(par['scaler_scale'])
        saved = predictions[predictions.model_year == year].set_index(['permno','target_month'])
        for model in cfg['models']:
            reconstructed = x @ np.array(par['models'][model]['coef']) + par['target_train_mean']
            observed = saved.loc[pd.MultiIndex.from_frame(rows[['permno','target_month']]), model].to_numpy()
            max_error = max(max_error, float(np.max(np.abs(observed-reconstructed))))
    if max_error > 1e-10: raise ValueError('parameter reconstruction mismatch')
    predictions.to_csv(out/'predictions.csv', index=False)
    pd.DataFrame({'permno':selected}).to_csv(out/'selected_securities.csv',index=False)
    pd.DataFrame([{'model':m,'target_month':str(month), **prediction_metrics(g.ret_exc_lead1m,g[m])}
                  for month,g in predictions.groupby('target_month') for m in cfg['models']]).to_csv(out/'monthly_metrics.csv',index=False)
    write_json(out/'config.json',cfg)
    write_json(out/'parameters.json',params_by_year)
    scope = ('all available monthly numerical observations; prediction experiment, no portfolio backtest'
             if cfg.get('universe') == 'all_available' else 'initial-universe smoke experiment')
    write_json(out/'results.json',{'scope':scope,'preprocessing':diagnostics,'annual':annual,'pooled_test':pooled})
    coverage = predictions.groupby('target_month').agg(prediction_rows=('permno','size'), finite_labels=('label_available','sum'))
    coverage['missing_labels'] = coverage.prediction_rows - coverage.finite_labels
    coverage.to_csv(out/'monthly_coverage.csv')
    write_json(out/'verification.json',{'parameter_reconstruction_max_error':max_error,'prediction_rows':len(predictions),'test_months':len(expected),'unique_keys':True,'all_predictions_finite':True})
    source_files = [args.config,__file__,'scripts/run_member_d.py','src/models/official_linear_baseline.py','src/data/prepare_quant.py','src/data/build_samples.py','src/data/splits.py','src/utils/dates.py','src/training/metrics.py','requirements-member-d.txt','scripts/verify_official_source.py','tests/test_official_baseline.py']
    outputs = [p.name for p in out.iterdir() if p.is_file()]
    write_json(out/'manifest.json',{'command':shlex.join([sys.executable,'-m','scripts.run_official_member_d','--config',args.config]),'cwd':str(Path.cwd()),'python':platform.python_version(),
               'packages':{m.__name__:m.__version__ for m in [np,pd,sklearn,pyarrow]},'elapsed_seconds':time.monotonic()-start,
               'input_sha256':input_hashes,
               'source_sha256':{str(p):sha256(p) for p in source_files},'output_sha256':{p:sha256(out/p) for p in outputs},
               'grid_sizes':{k:len(v) for k,v in official_grids().items()}})
    print(json.dumps(pooled,indent=2),flush=True)

if __name__ == '__main__': main()

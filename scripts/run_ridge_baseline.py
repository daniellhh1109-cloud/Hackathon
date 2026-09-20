"""Reproduce the small baseline baseline. Run from repository root as a module."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shlex
import sys
import time
import numpy as np
import pandas as pd
import pyarrow
import sklearn
from src.data.prepare_quant import load_factors, monthly_rank_transform
from src.data.build_samples import build_baseline_panel
from src.data.splits import annual_split
from src.models.linear_baseline import fit_year
from src.training.metrics import prediction_metrics


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/ridge_smoke.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config['preprocessing'] != 'full_same_month_median_then_average_rank_minus1_plus1' or config['model'] != 'Ridge_train_only_svd_with_intercept':
        raise ValueError('unsupported preprocessing/model configuration')
    years = config['test_years']
    if not years or len(set(years)) != len(years):
        raise ValueError('test_years must be nonempty and unique')
    for year in years:
        annual_split(year)
    factors = load_factors(config['factor_list'])
    if factors != load_factors(config['verified_factors']):
        raise ValueError('official and B-verified factor lists differ')
    start = time.monotonic()
    columns = list(dict.fromkeys(['permno', 'date', 'eom', 'ret_exc_lead1m', 'beta_60m', 'me', 'prc'] + factors))
    print('Loading numerical panel...', flush=True)
    raw = pd.read_parquet(config['chars'], columns=columns)
    # Only read feature months needed for this run into the preprocessing stage.
    last_target = annual_split(max(years)).test_end
    raw = raw[pd.to_datetime(raw['eom']) < pd.Timestamp(last_target)]
    panel = build_baseline_panel(raw)
    del raw
    selection_month = pd.Timestamp(config['selection_month']).date()
    if selection_month != pd.Timestamp('2015-01-31').date():
        raise ValueError('smoke selection must use initial 2015-01 cross-section')
    count = config['n_securities']
    if not isinstance(count, int) or isinstance(count, bool) or count < 2:
        raise ValueError('n_securities must be an integer >= 2')
    selected = sorted(panel.loc[panel.eom == selection_month, 'permno'].unique())[:count]
    if len(selected) != count:
        raise ValueError('not enough securities in selection month')
    print(f'Ranking {len(panel):,} rows / 147 factors across full monthly universes...', flush=True)
    features, diagnostics = monthly_rank_transform(panel, factors)
    mask = panel.permno.isin(selected)
    features = features.loc[mask].reset_index(drop=True)
    panel = panel.loc[mask].reset_index(drop=True)
    out = Path(config['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'permno': selected}).to_csv(out / 'selected_securities.csv', index=False)
    write_json(out / 'config.json', config)
    outputs, annual = [], []
    for year in years:
        print(f'Training {year}: train-only coefficients; validation alpha search...', flush=True)
        predictions, result, model = fit_year(panel, features, factors, year, config['alphas'])
        # Raw risk fields are explicitly separate from transformed model features.
        predictions = predictions.merge(panel[['permno', 'eom', 'beta_60m', 'me', 'prc']],
                                        on=['permno', 'eom'], how='left', validate='one_to_one')
        outputs.append(predictions)
        result['official_boundaries'] = asdict(annual_split(year))
        annual.append(result)
        pd.DataFrame({'variable': factors, 'coefficient': model.coef_}).to_csv(out / f'coefficients_{year}.csv', index=False)
        write_json(out / f'model_{year}.json', {'intercept': float(model.intercept_), 'alpha': model.alpha, 'features': factors})
    prediction = pd.concat(outputs, ignore_index=True).sort_values(['target_month', 'permno'])
    if prediction.duplicated(['permno', 'target_month']).any():
        raise ValueError('duplicate prediction key')
    expected_months = {str(d.date()) for year in years for d in pd.date_range(
        annual_split(year).test_start, annual_split(year).test_end, freq='MS')}
    if set(prediction.target_month.astype(str)) != expected_months:
        raise ValueError('missing prediction months')
    prediction.to_csv(out / 'predictions.csv', index=False)
    monthly = [dict(target_month=str(month), **prediction_metrics(g.ret_exc_lead1m, g.prediction))
               for month, g in prediction.groupby('target_month')]
    pd.DataFrame(monthly).to_csv(out / 'monthly_metrics.csv', index=False)
    result = {'scope': 'small fixed initial-universe smoke test; not competition performance',
              'preprocessing': diagnostics, 'selected_securities': [int(x) for x in selected],
              'sample_rows': len(panel), 'annual': annual,
              'pooled_test': prediction_metrics(prediction.ret_exc_lead1m, prediction.prediction)}
    write_json(out / 'results.json', result)
    source_paths = [Path(args.config), Path(__file__), *[Path(p) for p in [
        'src/data/prepare_quant.py', 'src/data/build_samples.py', 'src/data/splits.py',
        'src/utils/dates.py', 'src/models/linear_baseline.py', 'src/training/metrics.py']]]
    manifest = {'command': shlex.join([sys.executable, '-m', 'scripts.run_ridge_baseline', '--config', args.config]),
                'cwd': str(Path.cwd()), 'python': platform.python_version(),
                'packages': {m.__name__: m.__version__ for m in [np, pd, pyarrow, sklearn]},
                'input_sha256': {config[k]: sha256(config[k]) for k in ['chars', 'factor_list', 'verified_factors']},
                'source_sha256': {str(p): sha256(p) for p in source_paths},
                'output_sha256': {p.name: sha256(p) for p in sorted(out.iterdir()) if p.name not in ['manifest.json', 'run.log'] and p.is_file()},
                'elapsed_seconds': round(time.monotonic() - start, 3)}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps(result['pooled_test'], indent=2), flush=True)
    print(f'Artifacts saved to {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()

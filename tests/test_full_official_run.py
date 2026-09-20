import json
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import run_official_member_d as runner
from src.models.official_linear_baseline import fit_official_year


def test_full_universe_retains_later_entrants_and_missing_labels():
    panel = pd.DataFrame({'permno': [1, 2], 'eom': pd.to_datetime(
        ['2015-01-31', '2025-01-31']).date, 'ret_exc_lead1m': [0.1, np.nan]})
    assert runner.universe_mask(panel, {'universe': 'all_available'}).all()
    with pytest.raises(ValueError):
        runner.universe_mask(panel, {'universe': 'all_available', 'n_securities': 128})


def test_full_pipeline_all_six_years(tmp_path, monkeypatch):
    # Small artificial cross-section, real preprocessing/fits/export/reconstruction.
    # Narrow grids keep this an integration test, not a performance experiment.
    factors = [f'f{i}' for i in range(147)]
    rng = np.random.default_rng(42)
    rows = []
    for eom in pd.date_range('2015-01-31', '2026-07-31', freq='ME'):
        for permno in range(1, 7 if eom.year >= 2022 else 6):
            rows.append(dict(permno=permno, date=eom, eom=eom,
                ret_exc_lead1m=np.nan if permno == 6 else rng.normal(0, .03),
                beta_60m=1., me=1000., prc=10.,
                **dict(zip(factors, rng.normal(size=147)))))
    chars = tmp_path/'chars.parquet'
    pd.DataFrame(rows).to_parquet(chars)
    factor_list = tmp_path/'factors.csv'
    pd.DataFrame({'variable': factors}).to_csv(factor_list, index=False)
    cfg = json.loads(open('configs/member_d_official_full.json').read())
    cfg.update(chars=str(chars), factor_list=str(factor_list), output_dir=str(tmp_path/'out'))
    config = tmp_path/'config.json'
    config.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, 'argv', ['runner', '--config', str(config)])
    grids = {'ols': [None], 'lasso': [.01], 'ridge': [1.], 'en': [.01]}
    monkeypatch.setattr(runner, 'fit_official_year',
        lambda p, x, f, y: fit_official_year(p, x, f, y, grids))
    runner.main()
    out = tmp_path/'out'
    predictions = pd.read_csv(out/'predictions.csv')
    assert predictions.target_month.nunique() == 68
    assert set(predictions.model_year) == set(range(2021, 2027))
    assert len(predictions[predictions.permno == 6]) > 0
    assert predictions.loc[predictions.permno == 6, 'ret_exc_lead1m'].isna().all()
    assert np.isfinite(predictions[cfg['models']]).all().all()
    assert json.loads((out/'verification.json').read_text())['parameter_reconstruction_max_error'] < 1e-10
    for year in range(2021, 2027):
        assert (out/f'parameters_{year}.json').exists()
    with pytest.raises(ValueError, match='not empty'):
        runner.main()


def test_coverage_rejects_missing_security_even_when_month_exists():
    months = pd.date_range('2021-01-01', '2021-12-01', freq='MS').date
    panel = pd.DataFrame([(p, m) for p in [1, 2] for m in months],
                         columns=['permno', 'target_month'])
    prediction = panel.iloc[:-1].assign(ols=0.)
    with pytest.raises(ValueError, match='universe'):
        runner.check_predictions(prediction, panel, [2021], ['ols'])

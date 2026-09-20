"""Regressions found while reconciling remote main and the complete workflow."""
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest
from test_monthly_inference import assets
from scripts.project import prepare


def test_reused_store_checks_current_source(assets):
    store, _, root = assets
    config = dict(store_dir=str(store), raw_path=str(root/'raw.parquet'),
                  factor_path=str(root/'factors.csv'))
    assert prepare(config)['reused_verified_store']
    frame = pd.read_parquet(config['raw_path'])
    frame.loc[0, 'f0'] += 1
    frame.to_parquet(config['raw_path'], index=False)
    with pytest.raises(ValueError, match='source differs'):
        prepare(config)


def test_reused_store_checks_factor_order(assets):
    store, _, root = assets
    factors = root/'factors.csv'
    frame = pd.read_csv(factors)
    frame.iloc[::-1].to_csv(factors,index=False)
    with pytest.raises(ValueError, match='factor order'):
        prepare(dict(store_dir=str(store),raw_path=str(root/'raw.parquet'),factor_path=str(factors)))


def test_submission_percentage_units_and_cli(tmp_path):
    # A 200%-gross book: 100 long + 100 short, each 1% of NAV.
    frame = pd.DataFrame({'Date':['2021-01-01']*200,'PERMNO':range(1,201),
                          'TICKER':['SYN']*200,'COMPANY NAME':['Synthetic']*200,
                          'WEIGHT':[1.]*100+[-1.]*100})
    path=tmp_path/'holdings.csv';frame.to_csv(path,index=False)
    run=subprocess.run([sys.executable,'-m','check_constraints',str(path)],
                       capture_output=True,text=True)
    assert run.returncode == 0, run.stderr
    report=json.loads(run.stdout)
    assert report['months'][0]['gross'] == pytest.approx(2)


def test_report_rejects_ambiguous_old_units_without_creating_directory(tmp_path):
    from src.portfolio.reporting import build_report
    (tmp_path/'run_result.json').write_text(json.dumps({'status':'completed'}))
    pd.DataFrame({'Date':['2021-01-01']}).to_csv(tmp_path/'timeline.csv',index=False)
    (tmp_path/'metrics.json').write_text('{}')
    pd.DataFrame({'WEIGHT':[.01]}).to_csv(tmp_path/'monthly_holdings.csv',index=False)
    with pytest.raises(ValueError,match='unit'):
        build_report(tmp_path)
    assert not (tmp_path/'report').exists()


def test_workflow_export_units_match_committed_weights(tmp_path,monkeypatch):
    # Run the actual optimizer and accounting on two synthetic months, not mocked outputs.
    import scripts.demo_pipeline as demo
    from src.portfolio.workflow import run_backtest
    def short_run(config, output):
        config['end_month']='2021-02-01'
        return run_backtest(config, output)
    monkeypatch.setattr(demo,'run_backtest',short_run)
    demo.main(['--output-dir',str(tmp_path/'demo')])
    root=tmp_path/'demo/run'
    csv=pd.read_csv(root/'report/Monthly_Holdings.csv',dtype={'PERMNO':str})
    committed=json.loads((root/'2021-01/holdings.json').read_text())
    actual=csv.loc[csv.Date.eq('2021-01-01')].set_index('PERMNO').WEIGHT
    for row in committed:
        assert actual[row['permno']] == pytest.approx(100*row['weight'])
    assert actual.abs().sum() == pytest.approx(200)
    assert json.loads((root/'run_result.json').read_text())['holdings_weight_unit']=='percent_of_NAV'

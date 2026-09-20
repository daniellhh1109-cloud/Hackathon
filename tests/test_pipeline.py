from dataclasses import asdict,replace
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.agent.tools import AgentPolicy,PortfolioTools
from src.agent.demo import components
from src.agent.controllers import MockController
from src.agent.portfolio_agent import run_month
from src.portfolio.optimizer import optimize_weights
from src.portfolio.context import ContextStore
from src.portfolio.backtest import OutcomeStore,read_benchmarks,summarize


def candidates():
    p,c,v=components();lookup={r['permno']:r for r in c}
    return [dict(lookup[r['permno']],predicted_excess_return=r['predicted_excess_return'],side='long' if i<150 else 'short')
            for i,r in enumerate(p[:300])]


def test_real_optimizer_active_count_and_exits():
    result=optimize_weights(candidates=candidates(),previous_weights=[{'permno':'EXIT','weight':.04}],policy=asdict(AgentPolicy()))
    assert result['status']=='optimal'
    assert result['constraint_report']['holdings_count']==300
    assert result['exit_traded_notional']==.04
    assert sum(abs(r['weight']) for r in result['weights'])==pytest.approx(2,abs=1e-8)
    assert min(abs(r['weight']) for r in result['weights'])>=.001-1e-8


@pytest.mark.parametrize('mutate',[lambda c:c.__setitem__(0,dict(c[0],beta_60m=float('nan'))),
    lambda c:c.append(c[0]),lambda c:c.__setitem__(0,dict(c[0],side='bad'))])
def test_bad_optimizer_input(mutate):
    c=candidates();mutate(c)
    assert optimize_weights(candidates=c,previous_weights=[],policy=asdict(AgentPolicy()))['status']=='failed'


def test_infeasible_optimizer_is_not_relaxed():
    policy=asdict(AgentPolicy());policy['gross_target']=.1
    result=optimize_weights(candidates=candidates(),previous_weights=[],policy=policy)
    assert result['status']=='failed' and result['failure_reason']=='infeasible'


def test_monthly_pnl_and_drift_after_commit(tmp_path):
    p,c,v=components();t=PortfolioTools('2021-01-01',p,c,v,optimize_weights)
    run_month(t,MockController(),tmp_path/'month')
    rows=json.loads((tmp_path/'month/holdings.json').read_text())
    returns=pd.DataFrame([{'permno':int(r['permno']),'eom':pd.Timestamp('2021-01-31'),'ret':.01} for r in rows])
    returns.to_parquet(tmp_path/'returns.parquet')
    result,drift=OutcomeStore(tmp_path/'returns.parquet').evaluate(tmp_path/'month',{},.001,transaction_cost_bps=10)
    net=sum(r['weight'] for r in rows)
    expected=.01*net+(1-net)*.001-.002
    assert result['total_return']==pytest.approx(expected,abs=1e-8)
    assert drift[rows[0]['permno']]==pytest.approx(rows[0]['weight']*1.01/(1+expected))


def test_missing_held_return_audit(tmp_path):
    p,c,v=components();run_month(PortfolioTools('2021-01-01',p,c,v,optimize_weights),MockController(),tmp_path/'month')
    pd.DataFrame({'permno':[1],'eom':[pd.Timestamp('2021-01-31')],'ret':[.1]}).to_parquet(tmp_path/'ret.parquet')
    with pytest.raises(ValueError,match='no imputation'):
        OutcomeStore(tmp_path/'ret.parquet').evaluate(tmp_path/'month',{},0)
    assert len(pd.read_csv(tmp_path/'month/missing_held_returns.csv'))==300


def test_outcomes_not_read_before_commit(tmp_path,monkeypatch):
    monkeypatch.setattr(pd,'read_parquet',lambda *a,**k:pytest.fail('outcomes read early'))
    with pytest.raises(FileNotFoundError):OutcomeStore('unused').evaluate(tmp_path,{},0)


def test_benchmark_and_market_gaps(tmp_path):
    pd.DataFrame({'observation_date':['2020-12-01','2021-01-01','2021-03-01'],'TB3MS':[1.2,2.4,3.6]}).to_csv(tmp_path/'tb.csv',index=False)
    pd.DataFrame({'observation_date':['2020-12-31','2021-01-29','2021-03-31'],'SP500':[100,110,121]}).to_csv(tmp_path/'sp.csv',index=False)
    b=read_benchmarks(tmp_path/'tb.csv',tmp_path/'sp.csv')
    assert b.loc['2021-01-01','benchmark']==pytest.approx(.002+.04/12)
    assert b.loc['2021-01-01','market']==pytest.approx(.1)
    assert pd.isna(b.loc['2021-03-01','market'])


def test_metrics_drawdown_includes_initial_nav():
    t=pd.DataFrame({'Date':['2021-01-01','2021-02-01'],'total_return':[-.1,.05],
                    'rf':[0,0],'TB3MS':[0,0],'benchmark':[.04/12]*2,'market':[.01,.02]})
    m,r=summarize(t)
    assert m['max_drawdown']==pytest.approx(.1)
    assert r.drawdown.iloc[0]==pytest.approx(-.1)


def test_setup_portable_no_overwrite(tmp_path):
    command=[sys.executable,'MAIN.py','setup','--config',str(tmp_path/'local/project.yaml'),'--device','cpu']
    first=subprocess.run(command,capture_output=True,text=True)
    assert first.returncode==0,first.stderr
    second=subprocess.run(command,capture_output=True,text=True)
    assert second.returncode!=0
    assert 'data/raw' in (tmp_path/'local/project.yaml').read_text()

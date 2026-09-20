"""baseline: evaluate committed weights, calculate prior drift and summarize risk."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.agent.tools import digest
from src.portfolio.metrics import monthly_total_return,traded_notional,performance


class OutcomeStore:
    """Reads only the requested realized month AFTER its holdings commit marker."""
    def __init__(self,path):self.path=path

    def evaluate(self,month_dir,previous,rf,*,transaction_cost_bps=10,annual_borrow_bps=0):
        d=Path(month_dir);commit=json.loads((d/'result.json').read_text())
        rows=json.loads((d/'holdings.json').read_text())
        if commit['status']!='committed' or digest(rows)!=commit['holdings_sha256']:
            raise ValueError('uncommitted or changed holdings')
        month=pd.Period(commit['target_month'],freq='M')
        # Parquet predicate keeps outcome access scoped to the locked decision month.
        frame=pd.read_parquet(self.path,columns=['permno','eom','ret'],
                              filters=[('eom','>=',month.start_time),('eom','<=',month.end_time)])
        if frame.permno.duplicated().any():raise ValueError('duplicate realized return')
        returns={str(int(r.permno)):r.ret for r in frame.itertuples()}
        weights={r['permno']:r['weight'] for r in rows}
        gaps=[{'permno':i,'weight':w,'month':str(month),'reason':'absent_or_nonfinite_total_return'}
              for i,w in weights.items() if w!=0 and (i not in returns or not np.isfinite(returns[i]))]
        if gaps:
            pd.DataFrame(gaps).to_csv(d/'missing_held_returns.csv',index=False)
            raise ValueError(f'{len(gaps)} held returns missing in {month}; see missing_held_returns.csv; no imputation or reselection')
        traded=traded_notional(weights,previous)
        pnl=monthly_total_return(weights,returns,rf,traded_notional=traded,
              transaction_cost_bps=transaction_cost_bps,annual_borrow_bps=annual_borrow_bps)
        growth=1+pnl['total_return']
        if growth<=0:raise ValueError('insolvent portfolio: stop simulation')
        drift={i:w*(1+float(returns[i]))/growth for i,w in weights.items() if w!=0}
        long_pnl=sum(w*float(returns[i]) for i,w in weights.items() if w>0)
        short_pnl=sum(w*float(returns[i]) for i,w in weights.items() if w<0)
        return dict(Date=month.start_time.date().isoformat(),**pnl,traded_notional=traded,
                    half_l1_turnover=traded/2,long_pnl=long_pnl,short_pnl=short_pnl),drift


def read_benchmarks(tb3ms_path,sp500_path):
    tb=pd.read_csv(tb3ms_path);sp=pd.read_csv(sp500_path)
    tb['Date']=pd.to_datetime(tb.iloc[:,0]).dt.to_period('M').dt.to_timestamp()
    tb['TB3MS']=pd.to_numeric(tb['TB3MS'],errors='raise')
    if tb.Date.duplicated().any():raise ValueError('duplicate TB3MS month')
    sp['date']=pd.to_datetime(sp.iloc[:,0]);sp['SP500']=pd.to_numeric(sp['SP500'],errors='coerce')
    sp=sp.sort_values('date').dropna(subset=['SP500'])
    if (sp.SP500<=0).any() or sp.date.duplicated().any():raise ValueError('invalid market levels')
    levels=sp.groupby(sp.date.dt.to_period('M')).SP500.last()
    market=levels/levels.shift(1)-1
    gaps=levels.index.astype('int64').to_series(index=levels.index).diff().ne(1)
    market[gaps]=np.nan
    table=tb.set_index('Date')[['TB3MS']]
    table['rf']=table.TB3MS/1200
    table['benchmark']=table.rf+.04/12
    market.index=market.index.to_timestamp()
    table['market']=market
    return table


def summarize(returns):
    import statsmodels.api as sm
    r=returns.copy()
    if len(r)<2:raise ValueError('at least two monthly observations required for report')
    metrics=performance(r.total_return.tolist(),r.rf.tolist(),r.TB3MS.tolist())
    metrics.pop('nav')
    metrics.update(mean_monthly_return=float(r.total_return.mean()),
                   annualized_arithmetic_return=float(r.total_return.mean()*12),
                   best_month_return=float(r.total_return.max()))
    if len(r)>=4 and r.market.std()>1e-12:
        model=sm.OLS(r.total_return-r.rf,sm.add_constant(r.market-r.rf)).fit(cov_type='HAC',cov_kwds={'maxlags':min(3,len(r)-2)})
        metrics.update(capm_alpha_monthly=float(model.params.iloc[0]),capm_alpha_t=float(model.tvalues.iloc[0]),
                       realized_beta=float(model.params.iloc[1]),beta_se=float(model.bse.iloc[1]),
                       market_correlation=float(r.total_return.corr(r.market)))
    else:
        metrics.update(capm_alpha_monthly=None,capm_alpha_t=None,realized_beta=None,beta_se=None,market_correlation=None)
    r['nav']=(1+r.total_return).cumprod();r['benchmark_nav']=(1+r.benchmark).cumprod()
    r['market_nav']=(1+r.market).cumprod()
    r['drawdown']=r.nav/r.nav.cummax().clip(lower=1)-1
    r['active_return']=r.total_return-r.benchmark
    r['rolling_12m_active_mean']=r.active_return.rolling(12).mean()
    r['rolling_12m_ir']=np.sqrt(12)*r.active_return.rolling(12).mean()/r.active_return.rolling(12).std().replace(0,np.nan)
    excess=r.total_return-r.rf;market_excess=r.market-r.rf
    r['rolling_12m_beta']=excess.rolling(12).cov(market_excess)/market_excess.rolling(12).var().replace(0,np.nan)
    # JSON does not silently encode NaN as a valid performance number.
    metrics={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in metrics.items()}
    return metrics,r

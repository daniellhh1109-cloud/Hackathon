"""Artificial 68-month end-to-end acceptance, including the real convex optimizer."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.portfolio.workflow import run_backtest
from src.portfolio.reporting import build_report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True)
    args=p.parse_args(argv);root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=False)
    inputs=root/'synthetic_inputs';inputs.mkdir();pred=inputs/'predictions';pred.mkdir()
    dates=pd.period_range('2020-12','2026-08',freq='M');panel=[]
    ids=np.arange(10000,10320)
    for j,period in enumerate(dates):
        for i,identifier in enumerate(ids):
            panel.append(dict(permno=int(identifier),eom=period.end_time.normalize(),date=period.end_time.normalize(),
                beta_60m=1.,me=10000.,dolvol=1e8,prc=30.,common=1,primary_sec=1,
                ticker=f'SYN{i}',company_name=f'Artificial Company {i}',ticker_name_reference_date=period.end_time.normalize(),
                ticker_name_status='verified_in_observation_month',ret=.003*np.cos(j+i/30),ret_exc_lead1m=.003*np.cos(j+1+i/30)-.001))
        if j:
            pd.DataFrame({'target_month':str(period),'permno':ids,'predicted_excess_return':(160-np.arange(320))/1e5,
                'rank':np.arange(1,321),'quant_end_month':str(period-1),'model_year':period.year,'checkpoint_sha256':'synthetic-fixture'})[['target_month','permno','predicted_excess_return','rank','quant_end_month','model_year','checkpoint_sha256']].to_parquet(pred/f'{period}.parquet',index=False)
    pd.DataFrame(panel).to_parquet(inputs/'panel.parquet',index=False)
    pd.DataFrame({'observation_date':[str(x.start_time.date()) for x in dates],'TB3MS':[1.2]*len(dates)}).to_csv(inputs/'tb.csv',index=False)
    pd.DataFrame({'observation_date':[str(x.end_time.date()) for x in dates],
                  'SP500':100*np.cumprod(1+.005+.01*np.sin(np.arange(len(dates))))}).to_csv(inputs/'sp.csv',index=False)
    config={'synthetic':True,'raw_path':str(inputs/'panel.parquet'),'predictions_dir':str(pred),
        'tb3ms_path':str(inputs/'tb.csv'),'sp500_path':str(inputs/'sp.csv'),
        'start_month':'2021-01-01','end_month':'2026-08-01','controller':{'kind':'mock'},
        'costs':{'transaction_cost_bps':10,'annual_borrow_bps':0}}
    result=run_backtest(config,root/'run');build_report(root/'run');print(result)
    return 0

if __name__=='__main__':raise SystemExit(main())

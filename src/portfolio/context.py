"""Member B: point-in-time portfolio inputs; never loads return columns."""
from pathlib import Path
import numpy as np
import pandas as pd

CONTEXT_COLUMNS=['permno','eom','date','beta_60m','me','dolvol','prc','common','primary_sec',
                 'ticker','company_name','ticker_name_reference_date','ticker_name_status']


class ContextStore:
    def __init__(self,path):
        self.data=pd.read_parquet(path,columns=CONTEXT_COLUMNS)
        self.data['eom']=pd.to_datetime(self.data.eom)
        if self.data.duplicated(['permno','eom']).any():raise ValueError('duplicate security/month context')

    def prepare(self,predictions,month,*,min_price=5.,min_market_cap=1e9,min_dollar_volume=1e7):
        period=pd.Period(month,freq='M');cutoff=(period-1).end_time.normalize()
        p=predictions.copy()
        if p.empty or p.permno.duplicated().any():raise ValueError('missing/duplicate predictions')
        if not p.target_month.eq(str(period)).all():raise ValueError('prediction month mismatch')
        if not p.quant_end_month.eq(str(period-1)).all() or not p.model_year.eq(period.year).all():
            raise ValueError('prediction provenance/timing mismatch')
        if not np.isfinite(p.predicted_excess_return).all() or p.checkpoint_sha256.isna().any():
            raise ValueError('invalid prediction/provenance')
        c=self.data[self.data.eom.eq(cutoff)]
        merged=p.merge(c,on='permno',how='left',validate='one_to_one',indicator=True)
        tests={'context_missing':merged['_merge'].ne('both'),
               'invalid_risk':~np.isfinite(merged.beta_60m),
               'price':merged.prc.abs().lt(min_price)|merged.prc.isna(),
               'market_cap':merged.me.mul(1e6).lt(min_market_cap)|merged.me.isna(),
               'liquidity':merged.dolvol.lt(min_dollar_volume)|merged.dolvol.isna(),
               'security_type':merged.common.ne(1)|merged.primary_sec.ne(1),
               'future_observation':pd.to_datetime(merged.date).gt(cutoff)}
        rejected=pd.Series(False,index=merged.index)
        for mask in tests.values():rejected|=mask
        good=merged[~rejected].copy()
        audit={'predicted':len(p),'eligible':len(good),'excluded':int(rejected.sum()),
               'reason_counts_nonexclusive':{k:int(v.sum()) for k,v in tests.items()},
               'filters':{'min_price':min_price,'min_market_cap_USD':min_market_cap,'min_monthly_dollar_volume_USD':min_dollar_volume},
               'timing_assumption':'provided month-end characteristics are available at month end; provider PIT limitations remain'}
        pred=[];context=[]
        for r in good.itertuples():
            pred.append(dict(permno=str(int(r.permno)),target_month=period.start_time.date().isoformat(),
                predicted_excess_return=float(r.predicted_excess_return),information_cutoff=cutoff.date().isoformat(),
                model_selection_end=f'{period.year-1}-12-31',checkpoint_id=str(r.checkpoint_sha256)))
            context.append(dict(permno=str(int(r.permno)),target_month=period.start_time.date().isoformat(),
                available_at=cutoff.date().isoformat(),beta_60m=float(r.beta_60m),
                market_cap=float(r.me*1e6),dollar_volume=float(r.dolvol)))
        # Names are for output only. They never reach the controller or eligibility filter.
        labels={}
        for r in good.itertuples():
            ref=pd.to_datetime(r.ticker_name_reference_date,errors='coerce')
            valid=(pd.notna(ref) and ref<=cutoff and str(r.ticker_name_status).startswith('verified_')
                   and pd.notna(r.ticker) and pd.notna(r.company_name))
            labels[str(int(r.permno))]={'TICKER':str(r.ticker) if valid else '',
                'COMPANY NAME':str(r.company_name) if valid else '', 'historical_name_verified':bool(valid)}
        return pred,context,labels,audit


def read_prediction(root,month):
    root=Path(root); year=month[:4]
    options=[root/f'{month}.parquet',root/year/f'{month}.parquet',
             root/year/month/'predictions.parquet',root/month/'predictions.parquet']
    found=[p for p in options if p.exists()]
    if len(found)!=1:raise ValueError(f'expect exactly one prediction file for {month}, found {len(found)}')
    return pd.read_parquet(found[0]),found[0]


def audit_prediction_coverage(root,months):
    """Validate all requested files before constructing the first portfolio."""
    from src.inference.predict_month import validate_predictions
    reports=[]
    for date in months:
        month=date[:7];table,path=read_prediction(root,month)
        hashes=table.checkpoint_sha256.unique()
        if len(hashes)!=1:raise ValueError('mixed checkpoint hashes in monthly predictions')
        # Exact expected universe is checked by inference export. Here check schema and provenance.
        report=validate_predictions(table,month=month,expected_permnos=table.permno.tolist(),
            model_year=int(month[:4]),checkpoint_sha256=hashes[0])
        report['path']=str(path);reports.append(report)
    return reports

import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def history_lengths(frame):
    s=frame[['permno','eom']].drop_duplicates().sort_values(['permno','eom']).copy()
    m=pd.to_datetime(s.eom).dt.year*12+pd.to_datetime(s.eom).dt.month
    breaks=s.permno.ne(s.permno.shift()) | m.diff().ne(1)
    s['consecutive_months']=s.groupby(breaks.cumsum()).cumcount()+1
    s['has_12_calendar_months']=s.consecutive_months.ge(12)
    return s

def run(a):
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    paths={'chars':Path(a.chars),'filings':Path(a.filings),'factors':Path(a.factors)}
    summary={'inputs':{k:{'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)} for k,p in paths.items()},'versions':{'python':platform.python_version(),'pandas':pd.__version__,'pyarrow':pa.__version__}}
    features=pd.read_csv(paths['factors']).variable.tolist()
    c=pd.read_parquet(paths['chars']); c['eom']=pd.to_datetime(c.eom); c['date']=pd.to_datetime(c.date)
    missing=sorted(set(features)-set(c.columns))
    summary['whitelist']={'count':len(features),'duplicates':len(features)-len(set(features)),'absent':missing,'contains_target':'ret_exc_lead1m' in features}
    if missing or len(features)!=147 or len(set(features))!=147 or 'ret_exc_lead1m' in features: raise ValueError(summary['whitelist'])
    pd.DataFrame({'variable':features}).to_csv(out/'verified_factors.csv',index=False)
    for name,path in [('chars',paths['chars']),('filings',paths['filings'])]:
        schema=pq.read_schema(path)
        pd.DataFrame([{'field':f.name,'arrow_type':str(f.type),'role':'feature' if name=='chars' and f.name in features else 'auxiliary_or_metadata'} for f in schema]).to_csv(out/f'{name}_schema.csv',index=False)
    c.isna().sum().rename('null_count').to_frame().assign(null_rate=c.isna().mean()).to_csv(out/'chars_missingness.csv',index_label='field')
    miss=c[features].isna(); rates=miss.groupby(c.eom).mean(); rates.to_csv(out/'factor_missingness_by_month.csv',index_label='eom')
    allmiss=rates.stack(); allmiss[allmiss.eq(1)].rename('missing_rate').to_csv(out/'all_missing_factor_months.csv')
    inf=pd.Series({f:int(np.isinf(c[f].to_numpy(dtype=float)).sum()) for f in features}); inf.to_csv(out/'factor_infinity.csv',index_label='factor',header=['infinite_count'])
    hist=history_lengths(c); hist.to_csv(out/'calendar_history.csv',index=False)
    gaps=c[['permno','eom']].sort_values(['permno','eom']).copy(); gaps['previous_eom']=gaps.groupby('permno').eom.shift(); gaps['calendar_gap']=(gaps.eom.dt.year-gaps.previous_eom.dt.year)*12+gaps.eom.dt.month-gaps.previous_eom.dt.month
    gaps[gaps.calendar_gap.gt(1)].to_csv(out/'month_gaps.csv',index=False)
    summary['chars']={'rows':len(c),'columns':len(c.columns),'securities':c.permno.nunique(),'eom_min':str(c.eom.min().date()),'eom_max':str(c.eom.max().date()),'source_date_min':str(c.date.min().date()),'source_date_max':str(c.date.max().date()),'duplicate_permno_eom':int(c.duplicated(['permno','eom']).sum()),'duplicate_gvkey_iid_eom':int(c.duplicated(['gvkey','iid','eom']).sum()),'null_keys':int(c[['permno','eom']].isna().any(axis=1).sum()),'non_month_end':int((c.eom!=c.eom+pd.offsets.MonthEnd(0)).sum()),'date_after_eom':int(c.date.gt(c.eom).sum()),'all_missing_factor_months':int(allmiss.eq(1).sum()),'infinite_factor_values':int(inf.sum()),'rows_with_12_month_history':int(hist.has_12_calendar_months.sum()),'securities_with_any_12_month_history':hist.loc[hist.has_12_calendar_months,'permno'].nunique(),'month_gaps':int(gaps.calendar_gap.gt(1).sum())}
    c.groupby(c.eom.dt.year).agg(rows=('permno','size'),securities=('permno','nunique'),source_crsp_mean=('source_crsp','mean'),last_observation=('date','max'),missing_target=('ret_exc_lead1m',lambda x:x.isna().sum())).to_csv(out/'chars_by_year.csv')
    # Stream all columns for null counts, retaining only compact join metadata.
    compact=[]; nulls={f.name:0 for f in pq.read_schema(paths['filings'])}; empty=0; hash_bad=0
    keep=['document_id','permno','gvkey','iid','filing_date','text_sha256','filing_time_precision','characteristics_link_basis','cusip_date_precision']
    for batch in pq.ParquetFile(paths['filings']).iter_batches(batch_size=4096):
        for name in nulls: nulls[name]+=batch.column(name).null_count
        for row in batch.select(['text','text_sha256']).to_pylist():
            t=row['text']; empty+=int(t is None or not t.strip()); hash_bad+=int(t is not None and hashlib.sha256(t.encode()).hexdigest()!=row['text_sha256'])
        compact.append(batch.select(keep).to_pandas())
    f=pd.concat(compact,ignore_index=True); f.filing_date=pd.to_datetime(f.filing_date); f['eom']=f.filing_date+pd.offsets.MonthEnd(0)
    pd.DataFrame({'null_count':nulls}).assign(null_rate=lambda x:x.null_count/len(f)).to_csv(out/'filings_missingness.csv',index_label='field')
    join=f.merge(c[['permno','gvkey','iid','eom']],how='left',on=['permno','gvkey','iid','eom'],indicator=True,validate='many_to_one')
    unmatched=join[join._merge.eq('left_only')]; unmatched.to_csv(out/'filings_without_same_month.csv',index=False)
    counts=f.groupby(['permno','eom']).size().rename('filing_count'); coverage=c[['permno','eom']].merge(counts,on=['permno','eom'],how='left',validate='one_to_one'); coverage.filing_count=coverage.filing_count.fillna(0).astype(int)
    coverage.groupby('eom').agg(stock_months=('permno','size'),with_filings=('filing_count',lambda x:x.gt(0).sum()),filing_count=('filing_count','sum')).to_csv(out/'filing_coverage_by_month.csv')
    summary['filings']={'rows':len(f),'columns':len(nulls),'securities':f.permno.nunique(),'date_min':str(f.filing_date.min().date()),'date_max':str(f.filing_date.max().date()),'empty_text':empty,'text_hash_mismatch':hash_bad,'duplicate_document_id':int(f.duplicated('document_id').sum()),'duplicate_security_date_hash':int(f.duplicated(['permno','filing_date','text_sha256']).sum()),'null_keys':int(f[['document_id','permno','filing_date']].isna().any(axis=1).sum()),'securities_absent_from_chars':len(set(f.permno)-set(c.permno)),'same_month_exact_link':len(f)-len(unmatched),'no_same_month_exact_link':len(unmatched),'stock_months_without_filings':int(coverage.filing_count.eq(0).sum()),'repeated_text_hash_rows':int(f.duplicated('text_sha256').sum())}
    for field in ['filing_time_precision','characteristics_link_basis','cusip_date_precision']: f[field].value_counts(dropna=False).to_csv(out/f'{field}.csv')
    f.groupby(f.filing_date.dt.year).size().to_csv(out/'filings_by_year.csv',header=['rows'])
    for keys,label in [(['permno','eom'],'chars'),(['document_id'],'filings_document')]:
        table=c if label=='chars' else f
        table.loc[table.duplicated(keys,keep=False),keys].to_csv(out/f'{label}_duplicates.csv',index=False)
    # Descriptive handoff example. Date-only eligibility is a proposal, not certified public availability.
    c.loc[c.permno.eq(12490)&c.eom.between('2020-01-01','2020-12-31'),['permno','gvkey','iid','date','eom','ret','ret_exc','ret_exc_lead1m']].assign(target_month=lambda x:x.eom+pd.offsets.MonthBegin(1)).to_csv(out/'ibm_2021_numeric_timeline.csv',index=False)
    f.loc[f.permno.eq(12490)&f.filing_date.between('2020-07-01','2021-01-31'),keep].assign(before_decision_date=lambda x:x.filing_date.lt('2021-01-01')).to_csv(out/'ibm_2021_filing_timeline.csv',index=False)
    if a.existing_outputs:
        for filename in ['missing_held_returns.csv','missing_historical_labels.csv']:
            p=Path(a.existing_outputs)/filename
            h=pd.read_csv(p,dtype={'gvkey':str,'iid':str}); h['eom_target']=pd.to_datetime(h.Date)+pd.offsets.MonthEnd(0)
            v=h[['permno','Date','eom_target']].merge(c[['permno','eom','ret','ticker','company_name','date']],left_on=['permno','eom_target'],right_on=['permno','eom'],how='left',indicator=True,validate='many_to_one')
            v['audit_status']=np.where(v._merge.eq('left_only'),'no_target_month_row',np.where(v.ret.isna(),'target_month_ret_null','return_present')) if filename.startswith('missing_held') else np.where(v._merge.eq('left_only'),'no_target_month_row',np.where(v.ticker.isna()|v.company_name.isna(),'historical_label_unverified','label_present'))
            v.to_csv(out/filename,index=False); summary[filename]={'input_sha256':sha(p),'rows':len(v),'status':v.audit_status.value_counts().to_dict()}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda v:int(v)),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda v:int(v)))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--chars',required=True); p.add_argument('--filings',required=True); p.add_argument('--factors',default='docs/week1/sources/factor_char_list.csv'); p.add_argument('--output',default='outputs/week1'); p.add_argument('--existing-outputs'); run(p.parse_args())

"""Independent event-boundary auditing and strict common-universe evaluation.

No model fitting or portfolio construction occurs here. Labels are joined only
when the caller supplies already frozen predictions for both models.
"""
import numpy as np
import pandas as pd
from src.training.evaluation import evaluate_table

KEYS=['permno','target_month']


def validate_keys(table, name):
    if not set(KEYS)<=set(table) or table.empty:
        raise ValueError(f'{name}: nonempty security/target keys required')
    data=table.copy()
    ids=pd.to_numeric(data.permno,errors='raise')
    if (data.permno.map(lambda x:isinstance(x,(bool,np.bool_))).any() or
        not np.isfinite(ids).all() or (ids<=0).any() or (ids%1!=0).any()):
        raise ValueError(f'{name}: invalid permno')
    if not data.target_month.map(lambda x:isinstance(x,str)).all() or not data.target_month.str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])').all():
        raise ValueError(f'{name}: target_month must be YYYY-MM')
    pd.PeriodIndex(data.target_month,freq='M')
    data['permno']=ids.astype('int64')
    if data.duplicated(KEYS).any():raise ValueError(f'{name}: duplicate security-target keys')
    return data


def _event_frame(events):
    required={'source_index','permno','filing_date','available_at_utc','status'}
    if not required<=set(events):raise ValueError('event evidence fields missing')
    data=events.copy()
    if data.source_index.isna().any() or data.source_index.duplicated().any():
        raise ValueError('duplicate/missing source event identity')
    allowed={'success','pending','failed','empty','invalid','duplicate'}
    if not data.status.isin(allowed).all():raise ValueError('unknown event status')
    if data.status.eq('invalid').any():
        raise ValueError('invalid source records require resolution before coverage certification')
    ids=pd.to_numeric(data.permno,errors='raise')
    if not np.isfinite(ids).all() or (ids<=0).any() or (ids%1!=0).any():raise ValueError('invalid event permno')
    data['permno']=ids.astype('int64')
    days=pd.to_datetime(data.filing_date,errors='raise')
    if days.isna().any() or not days.eq(days.dt.normalize()).all():raise ValueError('filing_date must be a calendar date')
    available=pd.to_datetime(data.available_at_utc,errors='raise')
    if available.isna().any():raise ValueError('event availability is missing')
    if len(data) and available.dt.tz is None:raise ValueError('event availability must be timezone aware')
    data['_available']=pd.to_datetime(data.available_at_utc,utc=True)
    data['_filing_month']=days.dt.to_period('M')
    if len(data) and (data['_available']<pd.to_datetime(days,utc=True)).any():
        raise ValueError('availability cannot precede provider filing date')
    return data


def event_coverage(keys, events):
    """Count eligible events independently of predictions/returns or encoding success.

    Six provider-filing months before the target, with availability strictly
    before target-month start. Pending eligible events are NOT no-event samples.
    """
    keys=validate_keys(keys,'coverage universe')[KEYS]
    events=_event_frame(events)
    rows=[]
    for month,samples in keys.groupby('target_month',sort=True):
        target=pd.Period(month,freq='M')
        cutoff=target.start_time.tz_localize('UTC')
        selected=events[(events['_filing_month']>=target-6)&(events['_filing_month']<target)&
                        (events['_available']<cutoff)&~events.status.eq('duplicate')]
        counts=selected.groupby(['permno','status']).size().unstack(fill_value=0)
        part=samples.copy()
        for status in ('success','pending','failed','empty'):
            series=counts[status] if status in counts else pd.Series(dtype='int64')
            part['events_'+status]=part.permno.map(series).fillna(0).astype('int64')
        part['event_count']=part[['events_success','events_pending','events_failed','events_empty']].sum(axis=1)
        part['has_recent_filing']=part.event_count.gt(0)
        part['text_complete']=part[['events_pending','events_failed','events_empty']].sum(axis=1).eq(0)
        rows.append(part)
    return pd.concat(rows,ignore_index=True)


def validate_selected_events(events, *, permno, target_month):
    """Reject an already-selected evidence set containing a future/wrong-stock event."""
    validate_keys(pd.DataFrame({'permno':[permno],'target_month':[target_month]}),'sample')
    data=_event_frame(events)
    target=pd.Period(target_month,freq='M')
    if not data.permno.eq(permno).all():raise ValueError('event belongs to another security')
    if not data.status.eq('success').all():raise ValueError('selected event is not successfully encoded')
    if not ((data['_filing_month']>=target-6)&(data['_filing_month']<target)).all():
        raise ValueError('event outside six-month filing window')
    if not (data['_available']<target.start_time.tz_localize('UTC')).all():
        raise ValueError('future or unavailable event at prediction cutoff')
    return True


def _same_universe(reference,other,name):
    check=reference[KEYS].merge(other[KEYS],on=KEYS,how='outer',indicator=True,validate='one_to_one')
    counts=check['_merge'].value_counts()
    if not check['_merge'].eq('both').all():
        raise ValueError(f'{name}: universe mismatch (missing={int(counts.get("left_only",0))}, extra={int(counts.get("right_only",0))})')


def compare_predictions(quant, multimodal, labels, coverage, *, huber_delta=1.0,
                        return_unit='decimal_excess_return'):
    if return_unit!='decimal_excess_return':raise ValueError('explicit decimal excess-return units required')
    coverage=validate_keys(coverage,'coverage')
    for column in ('text_complete','has_recent_filing'):
        if column not in coverage or coverage[column].dtype!=bool:
            raise ValueError(f'coverage requires boolean {column}')
    if not coverage.text_complete.all():raise ValueError('incomplete text coverage; do not treat pending filings as no events')
    frames=[]
    for name,frame in [('quant',quant),('multimodal',multimodal)]:
        frame=validate_keys(frame,name)
        if {'target','realized_target','ret_exc_lead1m','ret','ret_exc'}&set(frame):
            raise ValueError('prediction files must not contain realized labels')
        if 'predicted_excess_return' not in frame:raise ValueError('prediction column missing')
        values=pd.to_numeric(frame.predicted_excess_return,errors='raise')
        if not np.isfinite(values).all():raise ValueError('non-finite predictions')
        frame=frame[KEYS].assign(**{name:values.to_numpy()})
        _same_universe(coverage,frame,name)
        frames.append(frame)
    labels=validate_keys(labels,'labels')
    if 'realized_target' not in labels:raise ValueError('realized_target missing')
    labels=labels[KEYS+['realized_target']].copy()
    labels['realized_target']=pd.to_numeric(labels.realized_target,errors='raise')
    table=coverage.merge(frames[0],on=KEYS,validate='one_to_one').merge(frames[1],on=KEYS,validate='one_to_one')
    table=table.merge(labels,on=KEYS,how='left',validate='one_to_one')
    summaries=[];monthly=[]
    for group,selected in [('all',table),('with_events',table[table.has_recent_filing]),('without_events',table[~table.has_recent_filing])]:
        if selected.empty:
            for name in ('quant','multimodal'):
                summaries.append({'group':group,'model':name,'n_predictions':0,'n_scored':0,'status':'no_samples'})
            continue
        summary,by_month=evaluate_table(selected,['quant','multimodal'],huber_delta)
        summaries.extend({'group':group,'status':'scored',**r} for r in summary)
        monthly.extend({'group':group,**r} for r in by_month)
    return table,summaries,monthly


def diagnostic_summary(details):
    """Evaluate gates only on event-bearing samples; absent events have zero correction."""
    import torch
    gate=details['gate'].detach().cpu()
    valid=details['has_events'].detach().cpu()
    if not torch.isfinite(gate).all() or (gate<0).any() or (gate>1).any():raise ValueError('invalid gate values')
    selected=gate[valid]
    correction=details['text_correction'].detach().cpu()
    if torch.count_nonzero(correction[~valid]):raise ValueError('nonzero empty-event correction')
    return {'samples':len(valid),'with_events':int(valid.sum()),'without_events':int((~valid).sum()),
            'gate_with_events_mean':float(selected.mean()) if selected.numel() else None,
            'gate_with_events_min':float(selected.min()) if selected.numel() else None,
            'gate_with_events_max':float(selected.max()) if selected.numel() else None,
            'gate_below_005_fraction':float((selected<.05).float().mean()) if selected.numel() else None,
            'gate_above_095_fraction':float((selected>.95).float().mean()) if selected.numel() else None,
            'all_empty_correction_is_zero':True,
            'interpretation':'Diagnostic only. Attention/gate values are not causal explanations.'}

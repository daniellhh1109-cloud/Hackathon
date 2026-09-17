import numpy as np
import pandas as pd
import pytest
import torch
from src.training.multimodal_evaluation import event_coverage,validate_selected_events,compare_predictions,diagnostic_summary


def events(status='success',date='2020-12-20',available='2020-12-23T00:00:00Z',permno=1):
    return pd.DataFrame([dict(source_index=0,permno=permno,filing_date=date,available_at_utc=available,status=status)])


def keys():return pd.DataFrame({'permno':[1,2],'target_month':['2021-01']*2})


@pytest.mark.parametrize('status',['pending','failed','empty'])
def test_unencoded_is_not_absent(status):
    result=event_coverage(keys(),events(status)).set_index('permno')
    assert result.loc[1,'has_recent_filing'] and not result.loc[1,'text_complete']
    assert not result.loc[2,'has_recent_filing'] and result.loc[2,'text_complete']


@pytest.mark.parametrize('date,available,expected',[
    ('2020-07-01','2020-07-04T00:00:00Z',1),
    ('2020-06-30','2020-07-03T00:00:00Z',0),
    ('2020-12-29','2021-01-01T00:00:00Z',0),
    ('2021-01-01','2021-01-04T00:00:00Z',0)])
def test_calendar_and_availability_boundaries(date,available,expected):
    assert event_coverage(keys(),events(date=date,available=available)).event_count.sum()==expected
    if not expected:
        with pytest.raises(ValueError):validate_selected_events(events(date=date,available=available),permno=1,target_month='2021-01')


def test_wrong_security_injection():
    with pytest.raises(ValueError,match='another security'):
        validate_selected_events(events(permno=2),permno=1,target_month='2021-01')


def test_duplicate_identity_rejected():
    with pytest.raises(ValueError,match='identity'):event_coverage(keys(),pd.concat([events(),events()]))


def test_duplicate_status_not_double_counted():
    assert event_coverage(keys(),events(status='duplicate')).event_count.sum()==0


def test_naive_availability_rejected():
    with pytest.raises(ValueError,match='timezone'):event_coverage(keys(),events(available='2020-12-23'))


def fixtures():
    k=keys()
    return (k.assign(predicted_excess_return=[0.,0.]),k.assign(predicted_excess_return=[.1,.1]),
            k.assign(realized_target=[.1,-.1]),k.assign(has_recent_filing=[True,False],text_complete=True))


def test_hand_calculated_metrics_and_groups():
    table,summary,_=compare_predictions(*fixtures())
    rows={(r['group'],r['model']):r for r in summary}
    assert rows['all','quant']['oos_r2_zero']==pytest.approx(0)
    assert rows['all','multimodal']['oos_r2_zero']==pytest.approx(-1)
    assert rows['with_events','multimodal']['oos_r2_zero']==pytest.approx(1)
    assert rows['all','quant']['huber_loss']==pytest.approx(.005)


def test_missing_label_same_scoring_sample():
    q,m,l,c=fixtures();l=l.iloc[:1]
    _,summary,_=compare_predictions(q,m,l,c)
    assert [r['n_scored'] for r in summary[:2]]==[1,1]
    assert [r['n_predictions'] for r in summary[:2]]==[2,2]


@pytest.mark.parametrize('mutation',['missing','extra','duplicate','nan','label','incomplete'])
def test_bad_input_rejected(mutation):
    q,m,l,c=fixtures()
    if mutation=='missing':m=m.iloc[:1]
    if mutation=='extra':m=pd.concat([m,m.iloc[:1].assign(permno=3)])
    if mutation=='duplicate':m=pd.concat([m,m.iloc[:1]])
    if mutation=='nan':m.loc[0,'predicted_excess_return']=np.nan
    if mutation=='label':m['realized_target']=0
    if mutation=='incomplete':c.loc[0,'text_complete']=False
    with pytest.raises(ValueError):compare_predictions(q,m,l,c)


def test_units_rejected():
    with pytest.raises(ValueError,match='units'):compare_predictions(*fixtures(),return_unit='percent')


def test_empty_gate_diagnostic():
    detail=dict(gate=torch.tensor([[.5],[.8]]),has_events=torch.tensor([True,False]),text_correction=torch.zeros(2,128))
    assert diagnostic_summary(detail)['gate_with_events_mean']==.5
    detail['text_correction'][1,0]=1
    with pytest.raises(ValueError,match='empty-event'):diagnostic_summary(detail)


@pytest.mark.parametrize('mutation',['none','year','factors','cache','loss','future','synthetic'])
def test_checkpoint_contract(mutation):
    from copy import deepcopy
    from scripts.evaluate_multimodal import checkpoint_contract
    from src.training.trainer import annual_bounds
    bounds=annual_bounds(2021)
    q=dict(target_year=2021,annual_bounds=bounds,selection_metric='validation_huber_loss',
           data_audit={role:{'target_month_range':bounds[role]} for role in ('train','validation')},
           feature_names=['factor'],preprocessing_metadata={'version':1},training={'huber_delta':1.},
           model_metadata={'name':'QuantRegressor'})
    m=deepcopy(q);m['model_metadata']={'name':'MultimodalRegressor'}
    metadata={'synthetic':False,'identity':'abc'};m['integration_metadata']={'text':metadata.copy()}
    if mutation=='year':m['target_year']=2022
    if mutation=='factors':m['feature_names']=['other']
    if mutation=='cache':m['integration_metadata']['text']['identity']='wrong'
    if mutation=='loss':m['training']['huber_delta']=.1
    if mutation=='future':m['data_audit']['train']['target_month_range']=['2021-01','2021-02']
    if mutation=='synthetic':metadata['synthetic']=True;m['integration_metadata']['text']=metadata.copy()
    if mutation=='none':assert checkpoint_contract(q,m,year=2021,partition='test',text_metadata=metadata)==bounds['test']
    else:
        with pytest.raises(ValueError):checkpoint_contract(q,m,year=2021,partition='test',text_metadata=metadata)

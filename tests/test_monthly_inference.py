"""Member E: label-free operation, provenance, exact universe and serialization."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from src.data.quant_dataset import prepare_store,write_json
from src.models.quant_model import QuantRegressor,model_metadata
from src.inference.predict_month import MonthlyPredictor,validate_predictions,canonical_month
from src.training.trainer import annual_bounds


@pytest.fixture
def assets(tmp_path):
    torch.set_num_threads(2)
    factors=[f'f{i}' for i in range(147)]
    rows=[]
    for p in pd.period_range('2020-01','2020-12',freq='M'):
        for security in [1,2,3]:
            rows.append({'permno':security,'date':p.to_timestamp('M'),'eom':p.to_timestamp('M'),
                         'ret_exc_lead1m':np.nan,**{f:float(security) for f in factors}})
    raw=tmp_path/'raw.parquet';whitelist=tmp_path/'factors.csv'
    pd.DataFrame(rows).to_parquet(raw,index=False)
    pd.DataFrame({'variable':factors}).to_csv(whitelist,index=False)
    store=tmp_path/'store';prepare_store(raw,whitelist,store)
    metadata=json.loads((store/'metadata.json').read_text())
    torch.manual_seed(42)
    model=QuantRegressor()
    checkpoint={'format_version':1,'target_year':2021,'feature_names':factors,
                'model_metadata':model_metadata(model.config),'preprocessing_metadata':metadata,
                'annual_bounds':annual_bounds(2021),'data_audit':{
                    'train':{'target_month_range':['2016-01','2018-12']},
                    'validation':{'target_month_range':['2019-01','2020-12']}},
                'epoch':2,'validation_loss':.01,'model_state_dict':model.state_dict()}
    path=tmp_path/'best.pt';torch.save(checkpoint,path)
    write_json(tmp_path/'summary.json',{'best_epoch':2,'best_validation_loss':.01})
    return store,path,tmp_path


def test_no_labels_needed_and_batch_size_independent_export(assets):
    store,path,root=assets
    (store/'labels.npy').unlink()  # physically absent, not just mocked
    predictor=MonthlyPredictor(store,path,year=2021,batch_size=1)
    first,_=predictor.predict('2021-01')
    together=MonthlyPredictor(store,path,year=2021,batch_size=512)
    second,_=together.predict('2021-01')
    a=first.sort_values('permno').predicted_excess_return.to_numpy()
    b=second.sort_values('permno').predicted_excess_return.to_numpy()
    np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5)
    result=together.export(root/'predictions',month='2021-01')
    assert result['predictions']==3 and result['months']==1
    assert result['labels_accessed'] is False
    assert len(result['files_sha256'])==2
    with pytest.raises(FileExistsError): together.export(root/'predictions',month='2021-01')
    with pytest.raises(ValueError,match='no eligible'): together.predict('2021-02')


@pytest.mark.parametrize('fault',['duplicate','nan','missing','provenance','cutoff','rank','label_column'])
def test_validation_rejects_corruption(assets,fault):
    store,path,_=assets
    predictor=MonthlyPredictor(store,path,year=2021)
    table,_=predictor.predict('2021-01')
    if fault=='duplicate':table.loc[1,'permno']=table.loc[0,'permno']
    elif fault=='nan':table.loc[0,'predicted_excess_return']=np.nan
    elif fault=='missing':table=table.iloc[:2]
    elif fault=='provenance':table.loc[0,'checkpoint_sha256']='wrong'
    elif fault=='cutoff':table.loc[0,'quant_end_month']='2021-01'
    elif fault=='rank':table.loc[0,'rank']=100
    else:table['realized_target']=1.
    with pytest.raises(ValueError):validate_predictions(table,month='2021-01',expected_permnos=[1,2,3],
        model_year=2021,checkpoint_sha256=predictor.checkpoint_sha256)


@pytest.mark.parametrize('fault',['factor_order','scope','incomplete_selection','year'])
def test_checkpoint_guards(assets,fault):
    store,path,root=assets
    checkpoint=torch.load(path,weights_only=True)
    if fault=='factor_order':checkpoint['feature_names'].reverse()
    elif fault=='scope':checkpoint['data_audit']['train']['target_month_range'][1]='2021-01'
    elif fault=='incomplete_selection':checkpoint['epoch']=1
    else:checkpoint['target_year']=2022
    torch.save(checkpoint,path)
    with pytest.raises(ValueError):MonthlyPredictor(store,path,year=2021)


def test_actual_feature_dates_checked(assets):
    store,path,_=assets
    context=pd.read_parquet(store/'raw_context.parquet')
    context.loc[0,'eom']=pd.Timestamp('2021-01-31')
    context.to_parquet(store/'raw_context.parquet',index=False)
    # The same corrupt source date now fails earlier, at Dataset construction.
    with pytest.raises(ValueError,match='source date must be in feature month'):
        MonthlyPredictor(store,path,year=2021)


@pytest.mark.parametrize('month',['2021-1','2021-13','2021-01-01','foo'])
def test_month_format(month):
    with pytest.raises(ValueError):canonical_month(month)


def test_saved_export_checker_rejects_modified_file(assets):
    from scripts.check_predictions import check_export
    store,path,root=assets
    (store/'labels.npy').unlink()
    MonthlyPredictor(store,path,year=2021).export(root/'export',month='2021-01')
    assert check_export(root/'export',store,path)['passed']
    with pytest.raises(ValueError,match='coverage'):
        check_export(root/'export',store,path,require_full_year=True)
    csv=root/'export/2021-01.csv'
    csv.write_text(csv.read_text()+'\n')
    with pytest.raises(ValueError,match='hash mismatch'):
        check_export(root/'export',store,path)


def test_export_checker_requires_complete_input_hashes(assets):
    from scripts.check_predictions import check_export
    store,path,root=assets
    MonthlyPredictor(store,path,year=2021).export(root/'export',month='2021-01')
    p=root/'export/provenance.json'
    metadata=json.loads(p.read_text())
    metadata['input_artifacts_sha256']={}
    write_json(p,metadata)
    with pytest.raises(ValueError,match='inventory incomplete'):
        check_export(root/'export',store,path)

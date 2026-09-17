"""Integration fixtures are artificial: never used as financial results."""
from datetime import date
import json
import numpy as np
import pandas as pd
import pytest
import torch
from test_monthly_inference import assets
from src.data import precompute_text_embeddings as cache
from src.data.multimodal_dataset import EventStore,MultimodalDataset
from src.models.multimodal_model import MultimodalRegressor,model_metadata
from src.inference.predict_multimodal import predict_month


@pytest.fixture
def integration(assets,monkeypatch):
    store,path,root=assets
    class FixtureEncoder:
        def __init__(self,*a,**k):pass
        def encode(self,text):return np.full(384,len(text),dtype=np.float32),len(text),1
    monkeypatch.setattr(cache,'FrozenMiniLM',FixtureEncoder)
    rows=[dict(document_id=str(i),permno=p,filing_date=date.fromisoformat(day),text=text)
          for i,(p,day,text) in enumerate([
              (1,'2020-07-01','oldest'),(1,'2020-12-01','recent'),(1,'2020-12-02','two'),
              (1,'2020-12-29','not available until January'),(1,'2020-06-30','too old'),
              (2,'2020-11-01','different security')])]
    source=root/'events.parquet';pd.DataFrame(rows).to_parquet(source,index=False)
    directory=root/'cache';cache.run_pipeline(source,directory,cache.EmbeddingConfig(),model_cache_dir=root)
    events=EventStore(directory)
    checkpoint=torch.load(path,weights_only=True)
    model=MultimodalRegressor()
    checkpoint.update(model_state_dict=model.state_dict(),model_metadata=model_metadata(model.config),
                      selection_metric='validation_huber_loss',integration_metadata={'text':events.metadata})
    torch.save(checkpoint,path)
    return store,path,root,directory,events


def test_alignment_order_counts_and_no_labels(integration):
    store,_,_,_,events=integration
    (store/'labels.npy').unlink()
    ds=MultimodalDataset(store,events,year=2021,partition='test')
    row=ds[0]
    assert 'target' not in row and row['permno']==1
    assert row['filing_counts'].tolist()==[1,0,0,0,0,2]
    assert row['filings'].shape==(6,2,384)
    assert row['filings'][0,0,0]==6 and row['filings'][5,0,0]==6
    assert row['filings'][5,1,0]==3
    assert len(ds.evidence(0))==3
    assert ds[1]['filing_counts'].tolist()==[0,0,0,0,1,0]
    assert ds[2]['filings'].shape==(6,0,384)
    assert ds[2]['filing_counts'].sum()==0


@pytest.mark.parametrize('status',['pending','failed','empty'])
def test_incomplete_never_dropped(integration,status):
    store,_,_,_,events=integration
    events.events.loc[events.events.source_index.eq(0),'status']=status
    with pytest.raises(ValueError,match='incomplete text coverage'):
        MultimodalDataset(store,events,year=2021,partition='test')
    with pytest.raises(ValueError,match='incomplete text window'):events.tensors(1,'2021-01')


def test_future_pending_is_not_eligible(integration):
    store,_,_,_,events=integration
    events.events.loc[events.events.source_index.eq(3),'status']='pending'
    ds=MultimodalDataset(store,events,year=2021,partition='test')
    assert len(ds)==3 and ds[0]['filing_counts'].sum()==3


def test_monthly_export_without_label_file(integration):
    store,path,root,directory,_=integration
    (store/'labels.npy').unlink()
    first=predict_month(store,directory,path,year=2021,month='2021-01',output_dir=root/'one',batch_size=1)
    second=predict_month(store,directory,path,year=2021,month='2021-01',output_dir=root/'all',batch_size=3)
    a=pd.read_parquet(root/'one/predictions.parquet').sort_values('permno')
    b=pd.read_parquet(root/'all/predictions.parquet').sort_values('permno')
    np.testing.assert_allclose(a.predicted_excess_return,b.predicted_excess_return,atol=1e-6,rtol=1e-5)
    assert len(a)==3 and first['labels_accessed'] is False and second['exact_universe']
    with pytest.raises(FileExistsError):predict_month(store,directory,path,year=2021,month='2021-01',output_dir=root/'one')


@pytest.mark.parametrize('fault',['year','future_train','provenance','factors','selection','quant_metadata'])
def test_inference_rejects_wrong_checkpoint(integration,fault):
    store,path,root,directory,_=integration
    checkpoint=torch.load(path,weights_only=True)
    if fault=='year':checkpoint['target_year']=2022
    if fault=='future_train':checkpoint['data_audit']['train']['target_month_range'][1]='2021-01'
    if fault=='provenance':checkpoint['integration_metadata']['text']['cache_manifest_sha256']='0'*64
    if fault=='factors':checkpoint['feature_names'].reverse()
    if fault=='selection':checkpoint['epoch']=99
    if fault=='quant_metadata':checkpoint['preprocessing_metadata']['format_version']=2
    torch.save(checkpoint,path)
    with pytest.raises(ValueError):predict_month(store,directory,path,year=2021,month='2021-01',output_dir=root/'bad')
    assert not (root/'bad').exists()


def test_cache_tampering_rejected(integration):
    _,_,_,directory,_=integration
    path=directory/'filing_embeddings.parquet'
    with path.open('ab') as stream:stream.write(b'corruption')
    with pytest.raises(ValueError):EventStore(directory)


def test_factory_training_to_monthly_export(integration):
    from src.data.quant_dataset import prepare_store
    from src.training.multimodal_components import build_components
    from src.training.multimodal import fit_multimodal
    from src.training.trainer import TrainingConfig
    import yaml
    _,_,root,directory,_=integration
    factors=[f'f{i}' for i in range(147)]
    rows=[]
    for month in pd.period_range('2017-01','2020-12',freq='M'):
        for security in [1,2,3]:
            rows.append(dict(permno=security,date=month.to_timestamp('M'),eom=month.to_timestamp('M'),
                ret_exc_lead1m=.01*security,**{f:float(security) for f in factors}))
    raw=root/'long_raw.parquet';pd.DataFrame(rows).to_parquet(raw,index=False)
    store=root/'long_store';prepare_store(raw,root/'factors.csv',store)
    config=root/'multimodal_data.yaml'
    config.write_text(yaml.safe_dump(dict(store_dir=str(store),cache_dir=str(directory),model_config='configs/multimodal_model.yaml')))
    components=build_components(year=2021,config_path=config)
    assert components['train_dataset'].event_store is components['validation_dataset'].event_store
    assert max(components['train_dataset'].samples.target_month)<='2018-12'
    assert max(components['validation_dataset'].samples.target_month)<='2020-12'
    _,summary=fit_multimodal(**components,config=TrainingConfig(batch_size=16,max_epochs=1),target_year=2021,output_dir=root/'trained')
    (store/'labels.npy').unlink()
    report=predict_month(store,directory,root/'trained/best.pt',year=2021,month='2021-01',output_dir=root/'trained_forecast')
    assert report['rows']==3 and report['labels_accessed'] is False

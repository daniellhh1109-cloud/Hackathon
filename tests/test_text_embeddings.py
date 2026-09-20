from dataclasses import replace
from datetime import date
import hashlib
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from src.data import precompute_text_embeddings as mod


def source(tmp_path,rows=None):
    rows=rows or [
        dict(document_id='d1',permno=1,filing_date=date(2020,12,31),text='alpha beta'),
        dict(document_id='d1',permno=2,filing_date=date(2020,12,31),text='alpha beta'),
        dict(document_id='d2',permno=1,filing_date=date(2020,12,1),text='gamma'),
        dict(document_id='d3',permno=1,filing_date=date(2020,12,2),text='   ')]
    path=tmp_path/'source.parquet'
    pq.write_table(pa.Table.from_pylist(rows),path)
    return path

class FakeEncoder:
    calls=0
    def __init__(self,*a,**kw):pass
    def encode(self,text):
        FakeEncoder.calls+=1
        return np.full(384,len(text),dtype=np.float32),len(text),1


def test_chunking_complete_coverage_no_extra_tail():
    chunks=list(mod.chunk_ids(list(range(449)),224,32))
    assert [len(x) for x in chunks]==[224,224,65]
    assert chunks[1][0]==192 and chunks[-1][-1]==448
    assert set(sum(chunks,[]))==set(range(449))
    assert len(list(mod.chunk_ids(list(range(224)),224,32)))==1
    assert list(mod.chunk_ids([],224,32))==[]
    with pytest.raises(ValueError):list(mod.chunk_ids([1],2,2))


def test_masked_pool_excludes_padding():
    hidden=torch.tensor([[[2.,4.],[4.,8.],[999.,999.]]])
    torch.testing.assert_close(mod.masked_mean_pool(hidden,torch.tensor([[1,1,0]])),torch.tensor([[3.,6.]]))
    with pytest.raises(ValueError,match='no valid tokens'):
        mod.masked_mean_pool(hidden,torch.zeros(1,3))


def test_cache_resume_preserves_security_links_and_provenance(tmp_path,monkeypatch):
    monkeypatch.setattr(mod,'FrozenMiniLM',FakeEncoder);FakeEncoder.calls=0
    path=source(tmp_path);out=tmp_path/'cache';cfg=mod.EmbeddingConfig()
    first=mod.run_pipeline(path,out,cfg,model_cache_dir=tmp_path,max_documents=1)
    assert first['counts']==dict(success=2,pending=1,failed=0,empty=1,invalid=0,duplicate=0)
    assert not first['complete'] and FakeEncoder.calls==1
    rows=pq.read_table(out/'filing_embeddings.parquet').to_pylist()
    assert rows[0]['available_at_utc'].date()==date(2021,1,3)
    assert rows[0]['embedding']==rows[1]['embedding'] and rows[0]['permno']!=rows[1]['permno']
    assert rows[2]['embedding'] is None
    second=mod.run_pipeline(path,out,cfg,model_cache_dir=tmp_path)
    assert second['complete'] and not second['fully_usable']
    assert FakeEncoder.calls==2 and second['unique_vectors']==2
    third=mod.run_pipeline(path,out,cfg,model_cache_dir=tmp_path)
    assert FakeEncoder.calls==2 and third['last_run']['attempts']==0
    metadata=json.loads((out/'text_metadata.json').read_text())
    from src.training.multimodal import validate_provenance
    validate_provenance(metadata)
    assert metadata['cache_manifest_sha256']==mod.sha256_file(out/'manifest.json')
    with pytest.raises(ValueError,match='identity mismatch'):
        mod.run_pipeline(path,out,replace(cfg,overlap_tokens=0),model_cache_dir=tmp_path)


def test_failures_not_empty_and_explicit_retry(tmp_path,monkeypatch):
    class Broken(FakeEncoder):
        def encode(self,text):raise ValueError('bad tokens')
    path=source(tmp_path);out=tmp_path/'cache';cfg=mod.EmbeddingConfig()
    monkeypatch.setattr(mod,'FrozenMiniLM',Broken)
    result=mod.run_pipeline(path,out,cfg,model_cache_dir=tmp_path)
    assert result['counts']['failed']==3 and result['counts']['empty']==1 and not result['complete']
    monkeypatch.setattr(mod,'FrozenMiniLM',FakeEncoder);FakeEncoder.calls=0
    mod.run_pipeline(path,out,cfg,model_cache_dir=tmp_path)
    assert FakeEncoder.calls==0
    result=mod.run_pipeline(path,out,cfg,model_cache_dir=tmp_path,retry_failed=True)
    assert result['counts']['failed']==0 and result['counts']['success']==3


def test_exact_duplicates_and_conflicting_ids(tmp_path):
    row=dict(document_id='d1',permno=1,filing_date=date(2020,1,1),text='some text')
    path=source(tmp_path,[row,row.copy()]);db=mod.connect(tmp_path/'state.sqlite3')
    mod.index_source(db,path,mod.EmbeddingConfig())
    assert db.execute("SELECT status,duplicate_of FROM observations WHERE source_index=1").fetchone()==('duplicate',0)
    db.close()
    path=source(tmp_path,[row,dict(row,text='different')]);db=mod.connect(tmp_path/'conflict.sqlite3')
    with pytest.raises(ValueError,match='conflicting'):
        mod.index_source(db,path,mod.EmbeddingConfig())
    assert db.execute('SELECT COUNT(*) FROM observations').fetchone()[0]==0
    db.close()


def test_device_failure_persists_status_and_resumes(tmp_path,monkeypatch):
    class DeviceFailure(FakeEncoder):
        def encode(self,text):raise RuntimeError('device unavailable')
    monkeypatch.setattr(mod,'FrozenMiniLM',DeviceFailure)
    path=source(tmp_path);out=tmp_path/'cache'
    with pytest.raises(RuntimeError,match='device unavailable'):
        mod.run_pipeline(path,out,mod.EmbeddingConfig(),model_cache_dir=tmp_path)
    report=json.loads((out/'coverage_report.json').read_text())
    assert report['last_run']['interrupted']
    assert report['counts']['failed']==2 and report['counts']['pending']==1


def test_empty_source_exports_valid_empty_table(tmp_path):
    path=tmp_path/'empty.parquet'
    pq.write_table(pa.table({'document_id':pa.array([],type=pa.string()),'permno':pa.array([],type=pa.int64()),
        'filing_date':pa.array([],type=pa.date32()),'text':pa.array([],type=pa.string())}),path)
    report=mod.run_pipeline(path,tmp_path/'cache',mod.EmbeddingConfig(),model_cache_dir=tmp_path,audit_only=True)
    assert report['source_rows']==0
    assert pq.read_table(tmp_path/'cache/filing_embeddings.parquet').num_rows==0


@pytest.mark.parametrize('kwargs',[{'revision':'main'},{'chunk_tokens':0},{'overlap_tokens':224},{'availability_delay_days':-1}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):mod.EmbeddingConfig(**kwargs)


def test_reader_rejects_partial_and_detects_tampering(tmp_path,monkeypatch):
    monkeypatch.setattr(mod,'FrozenMiniLM',FakeEncoder)
    path=source(tmp_path);out=tmp_path/'cache'
    mod.run_pipeline(path,out,mod.EmbeddingConfig(),model_cache_dir=tmp_path,max_documents=1)
    with pytest.raises(ValueError,match='not ready'):
        mod.load_verified_cache(out)
    table,metadata,report=mod.load_verified_cache(out,allow_partial=True)
    assert table.num_rows==4
    with (out/'coverage_report.json').open('a') as stream:stream.write(' ')
    with pytest.raises(ValueError,match='checksum mismatch'):
        mod.load_verified_cache(out,allow_partial=True)


def test_same_text_different_provider_ids_deduplicated_within_security(tmp_path):
    row=dict(document_id='first',permno=1,filing_date=date(2020,1,1),text='same content')
    path=source(tmp_path,[row,dict(row,document_id='alias'),dict(row,permno=2)])
    db=mod.connect(tmp_path/'state.sqlite3');mod.index_source(db,path,mod.EmbeddingConfig())
    assert db.execute('SELECT status FROM observations ORDER BY source_index').fetchall()==[('pending',),('duplicate',),('pending',)]
    db.close()


def test_audit_pending_is_unattempted_not_a_join_error(tmp_path):
    path=source(tmp_path);out=tmp_path/'cache'
    report=mod.run_pipeline(path,out,mod.EmbeddingConfig(),model_cache_dir=tmp_path,audit_only=True)
    assert report['counts']['pending']==3 and report['coverage']==0 and not report['complete']
    db=mod.connect(out/'cache.sqlite3')
    assert db.execute('SELECT COUNT(*) FROM vectors').fetchone()[0]==0
    assert db.execute('SELECT COUNT(*) FROM failures').fetchone()[0]==0
    db.close()


def test_cleanup_export_preserves_original_error(tmp_path,monkeypatch,caplog):
    class Broken(FakeEncoder):
        def encode(self,text):raise RuntimeError('original device failure')
    monkeypatch.setattr(mod,'FrozenMiniLM',Broken)
    def broken_export(*args,**kwargs):raise OSError('disk full during cleanup')
    monkeypatch.setattr(mod,'export_cache',broken_export)
    with pytest.raises(RuntimeError,match='original device failure') as caught:
        mod.run_pipeline(source(tmp_path),tmp_path/'cache',mod.EmbeddingConfig(),model_cache_dir=tmp_path)
    assert any('disk full during cleanup' in note for note in caught.value.__notes__)
    assert 'preserving original error' in caplog.text


def test_staging_failure_leaves_previous_snapshot_unchanged(tmp_path,monkeypatch):
    path=source(tmp_path);out=tmp_path/'cache';cfg=mod.EmbeddingConfig()
    mod.run_pipeline(path,out,cfg,model_cache_dir=tmp_path,audit_only=True)
    before={name:mod.sha256_file(out/name) for name in mod.EXPORT_FILES}
    original=mod.json_write
    def fail_staging(path,value):
        if path.name=='coverage_report.json':raise OSError('staging write failed')
        return original(path,value)
    monkeypatch.setattr(mod,'json_write',fail_staging)
    with pytest.raises(OSError,match='staging write failed'):
        mod.run_pipeline(path,out,cfg,model_cache_dir=tmp_path,audit_only=True)
    assert before=={name:mod.sha256_file(out/name) for name in mod.EXPORT_FILES}
    assert not (out/'export_in_progress.json').exists()
    mod.load_verified_cache(out,allow_partial=True)


def test_partial_publication_is_rejected_then_recovered(tmp_path,monkeypatch):
    from pathlib import Path
    path=source(tmp_path);out=tmp_path/'cache';cfg=mod.EmbeddingConfig()
    mod.run_pipeline(path,out,cfg,model_cache_dir=tmp_path,audit_only=True)
    original=Path.replace
    def fail_publish(self,target):
        if self.name=='coverage_report.json' and self.parent.name.startswith('.export-'):
            raise OSError('publication interrupted')
        return original(self,target)
    monkeypatch.setattr(Path,'replace',fail_publish)
    with pytest.raises(OSError,match='publication interrupted'):
        mod.run_pipeline(path,out,cfg,model_cache_dir=tmp_path,audit_only=True)
    with pytest.raises(ValueError,match='export incomplete'):
        mod.load_verified_cache(out,allow_partial=True)
    monkeypatch.setattr(Path,'replace',original)
    mod.run_pipeline(path,out,cfg,model_cache_dir=tmp_path,audit_only=True)
    assert not (out/'export_in_progress.json').exists()
    mod.load_verified_cache(out,allow_partial=True)


@pytest.mark.parametrize('invalid',[np.full(384,np.nan),np.full(384,np.inf),np.zeros(383)])
def test_invalid_vectors_are_failed_not_clipped(tmp_path,monkeypatch,invalid):
    class Invalid(FakeEncoder):
        def encode(self,text):return invalid,2,1
    monkeypatch.setattr(mod,'FrozenMiniLM',Invalid)
    report=mod.run_pipeline(source(tmp_path),tmp_path/'cache',mod.EmbeddingConfig(),model_cache_dir=tmp_path)
    assert report['counts']['failed']==3 and report['counts']['success']==0


def test_other_process_cannot_read_identity_while_locked(tmp_path):
    import subprocess
    import sys
    from filelock import FileLock
    out=tmp_path/'cache';out.mkdir()
    # An incomplete identity would fail JSON parsing if the child entered the lock.
    (out/'cache_identity.json').write_text('{')
    code="""
import json, sys
from pathlib import Path
from filelock import FileLock, Timeout
root=Path(sys.argv[1])
try:
    with FileLock(str(root/'cache.lock'),timeout=0):
        json.loads((root/'cache_identity.json').read_text())
except Timeout:
    print('locked')
else:
    raise AssertionError('unexpected lock acquisition')
"""
    with FileLock(str(out/'cache.lock'),timeout=0):
        child=subprocess.run([sys.executable,'-c',code,str(out)],capture_output=True,text=True,timeout=10)
    assert child.returncode==0,child.stderr
    assert child.stdout.strip()=='locked'

"""Frozen MiniLM filing embeddings with audited, transactional, resumable caching.

Only the supplied text is encoded locally. No returns, titles or future metadata
are fed to the encoder. Source/security links survive content-level cache reuse.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as daytime, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import sqlite3
import time
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from filelock import FileLock
import yaml

VERSION = 'filing_embeddings_v1'
COLUMNS = ['document_id', 'permno', 'filing_date', 'text']
LOGGER = logging.getLogger(__name__)
EXPORT_FILES = ('filing_embeddings.parquet', 'coverage_report.json', 'manifest.json', 'text_metadata.json')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = 'nreimers/MiniLM-L6-H384-uncased'
    revision: str = '3276f0fac9d818781d7a1327b3ff818fc4e643c0'
    chunk_tokens: int = 224
    overlap_tokens: int = 32
    batch_size: int = 32
    device: str = 'cpu'
    cpu_threads: int = 2
    availability_delay_days: int = 3

    def __post_init__(self):
        if not self.model or not re.fullmatch('[0-9a-f]{40}', self.revision):
            raise ValueError('model and immutable 40-character revision required')
        for name in ('chunk_tokens','batch_size','cpu_threads'):
            if type(getattr(self,name)) is not int or getattr(self,name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.overlap_tokens) is not int or not 0 <= self.overlap_tokens < self.chunk_tokens:
            raise ValueError('overlap_tokens must be >=0 and <chunk_tokens')
        if type(self.availability_delay_days) is not int or self.availability_delay_days < 0:
            raise ValueError('availability_delay_days must be a nonnegative integer')
        if self.device not in ('cpu','cuda','mps'):
            raise ValueError('unsupported device')


def read_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw,dict) or set(raw) != {'source_path','cache_dir','model_cache_dir','encoder'}:
        raise ValueError('expected source_path/cache_dir/model_cache_dir/encoder config')
    return raw, EmbeddingConfig(**raw['encoder'])


def chunk_ids(ids, size, overlap):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('invalid chunk size/overlap')
    start = 0
    while start < len(ids):
        end = min(start + size, len(ids))
        yield ids[start:end]
        if end == len(ids):
            break
        start = end - overlap


def masked_mean_pool(hidden, attention_mask):
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError('hidden/mask shapes disagree')
    denominator = attention_mask.sum(1, keepdim=True)
    if (denominator == 0).any():
        raise ValueError('chunk has no valid tokens')
    return (hidden * attention_mask.unsqueeze(-1)).sum(1) / denominator


class FrozenMiniLM:
    def __init__(self, config, cache_dir, *, local_files_only=True):
        from transformers import AutoTokenizer, AutoModel
        self.config = config
        torch.set_num_threads(config.cpu_threads)
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision,
            cache_dir=str(cache_dir), local_files_only=local_files_only, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(config.model, revision=config.revision,
            cache_dir=str(cache_dir), local_files_only=local_files_only, trust_remote_code=False,
            attn_implementation='eager').to(config.device)
        self.model.requires_grad_(False)
        self.model.eval()
        if self.model.config.hidden_size != 384 or self.model.config.num_hidden_layers != 6:
            raise ValueError('expected 6-layer, 384-hidden MiniLM')
        self.special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        if config.chunk_tokens + self.special_tokens > self.model.config.max_position_embeddings:
            raise ValueError('chunk plus special tokens exceeds model position limit')

    def encode(self, text):
        if not isinstance(text,str) or not text.strip():
            raise ValueError('empty text')
        if any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('encoder unexpectedly unfrozen')
        self.model.eval()
        ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
        if not ids:
            raise ValueError('tokenizer produced no tokens')
        total, count, pending = np.zeros(384, dtype=np.float64), 0, []
        def consume(chunks):
            features = [self.tokenizer.prepare_for_model(x, add_special_tokens=True,
                        truncation=False, return_attention_mask=True) for x in chunks]
            batch = self.tokenizer.pad(features, padding=True, return_tensors='pt')
            if batch['input_ids'].shape[1] > self.model.config.max_position_embeddings:
                raise ValueError('tokenized batch exceeds model limit')
            batch = {k:v.to(self.config.device) for k,v in batch.items()}
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                values = masked_mean_pool(hidden,batch['attention_mask']).cpu().float().numpy()
            if values.shape != (len(chunks),384) or not np.isfinite(values).all():
                raise ValueError('invalid encoder output')
            return values.astype(np.float64).sum(0)
        for chunk in chunk_ids(ids,self.config.chunk_tokens,self.config.overlap_tokens):
            pending.append(chunk)
            if len(pending) == self.config.batch_size:
                total += consume(pending); count += len(pending); pending = []
        if pending:
            total += consume(pending); count += len(pending)
        return (total/count).astype(np.float32), len(ids), count


def source_batches(source):
    return pq.ParquetFile(source).iter_batches(batch_size=512, columns=COLUMNS)


def connect(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS observations (
            source_index INTEGER PRIMARY KEY, document_id TEXT, permno INTEGER,
            filing_date TEXT, available_at_utc TEXT, text_sha256 TEXT,
            status TEXT NOT NULL, reason TEXT, duplicate_of INTEGER);
        CREATE INDEX IF NOT EXISTS content_index ON observations(text_sha256);
        CREATE TABLE IF NOT EXISTS vectors (
            text_sha256 TEXT PRIMARY KEY, embedding BLOB NOT NULL,
            token_count INTEGER NOT NULL, chunk_count INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS failures (
            text_sha256 TEXT PRIMARY KEY, error_type TEXT NOT NULL, message TEXT NOT NULL);
    ''')
    return db


def index_source(db, source, config):
    if db.execute("SELECT 1 FROM meta WHERE key='indexed'").fetchone():
        return
    # One transaction: an interrupted audit rolls back, so the source is rescanned safely.
    identities, seen, index = {}, {}, 0
    with db:
        for batch in source_batches(source):
            for row in batch.to_pylist():
                doc, permno, filing_date, text = (row[k] for k in COLUMNS)
                status, reason, duplicate_of = 'pending', None, None
                digest = hashlib.sha256(text.encode('utf-8')).hexdigest() if isinstance(text,str) else None
                valid = isinstance(doc,str) and bool(doc.strip()) and type(permno) is int and permno > 0 and type(filing_date) is date
                if not valid:
                    status,reason = 'invalid','invalid_identifier_or_date'
                elif not isinstance(text,str) or not text.strip():
                    status,reason = 'empty','empty_or_missing_text'
                else:
                    identity=(doc,permno)
                    signature=(filing_date.isoformat(),digest)
                    if identity in identities and identities[identity] != signature:
                        raise ValueError('conflicting document_id/security observations in source')
                    identities[identity]=signature
                    key=(permno,*signature)
                    if key in seen:
                        status,reason,duplicate_of='duplicate','exact_duplicate_observation',seen[key]
                    else:
                        seen[key]=index
                date_str=filing_date.isoformat() if type(filing_date) is date else None
                available=(datetime.combine(filing_date + timedelta(days=config.availability_delay_days),daytime(),tzinfo=timezone.utc).isoformat()
                           if type(filing_date) is date else None)
                db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?)',
                    (index,doc if isinstance(doc,str) else None,permno if type(permno) is int else None,
                     date_str,available,digest,status,reason,duplicate_of))
                index += 1
        db.execute("INSERT INTO meta VALUES ('indexed',?)",(str(index),))


def identity_for(source, config):
    from importlib.metadata import version
    settings=asdict(config)
    # Batch/thread counts are operational; device and arithmetic are part of cache identity.
    settings.pop('batch_size'); settings.pop('cpu_threads')
    return {'pipeline_version':VERSION,'source_sha256':sha256_file(source),
            'source_rows':pq.ParquetFile(source).metadata.num_rows,'encoder':settings,
            'pooling':'attention_mask_mean_including_special_tokens_then_equal_chunk_mean',
            'dtype':'float32','torch_version':str(torch.__version__),
            'transformers_version':version('transformers'),'synthetic':False}


def encode_pending(db, source, encoder, *, max_documents=None, retry_failed=False):
    if max_documents is not None and (type(max_documents) is not int or max_documents < 1):
        raise ValueError('max_documents must be a positive integer or None')
    vectors={r[0] for r in db.execute('SELECT text_sha256 FROM vectors')}
    failed={r[0] for r in db.execute('SELECT text_sha256 FROM failures')}
    pending={r[0] for r in db.execute("SELECT DISTINCT text_sha256 FROM observations WHERE status='pending'")} - vectors
    if not retry_failed:
        pending -= failed
    stats={'new_vectors':0,'new_failures':0,'new_chunks':0,'new_tokens':0,'attempts':0}
    started=time.monotonic()
    if pending:
        for batch in source_batches(source):
            for row in batch.to_pylist():
                text=row['text']
                if not isinstance(text,str): continue
                digest=hashlib.sha256(text.encode('utf-8')).hexdigest()
                if digest not in pending: continue
                if max_documents is not None and stats['attempts'] >= max_documents: break
                pending.remove(digest); stats['attempts']+=1
                try:
                    embedding,tokens,chunks=encoder.encode(text)
                    embedding=np.asarray(embedding,dtype='<f4')
                    if embedding.shape != (384,):
                        raise ValueError('embedding shape must be [384]')
                    if not np.isfinite(embedding).all():
                        raise ValueError('embedding must contain only finite values')
                    if tokens<1 or chunks<1: raise ValueError('empty token/chunk count')
                except (ValueError, RuntimeError) as exc:
                    # Persist explicit failure, never substitute a zero vector or no-event flag.
                    with db:
                        db.execute('INSERT OR REPLACE INTO failures VALUES (?,?,?)',
                                   (digest,type(exc).__name__,str(exc)[:300]))
                    stats['new_failures']+=1
                    if isinstance(exc,RuntimeError):
                        # Device/OOM faults usually affect subsequent documents too; stop visibly.
                        raise
                    continue
                with db:
                    db.execute('INSERT INTO vectors VALUES (?,?,?,?)',(digest,embedding.tobytes(),int(tokens),int(chunks)))
                    db.execute('DELETE FROM failures WHERE text_sha256=?',(digest,))
                stats['new_vectors']+=1;stats['new_tokens']+=tokens;stats['new_chunks']+=chunks
                if stats['attempts'] % 10 == 0:
                    print(json.dumps({**stats,'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
            if not pending or (max_documents is not None and stats['attempts'] >= max_documents): break
    stats['elapsed_seconds']=time.monotonic()-started
    return stats


EXPORT_SCHEMA=pa.schema([
    ('source_index',pa.int64()),('document_id',pa.string()),('permno',pa.int64()),
    ('filing_date',pa.date32()),('available_at_utc',pa.timestamp('us',tz='UTC')),
    ('text_sha256',pa.string()),('status',pa.string()),('reason',pa.string()),
    # Variable list permits null vectors in partial caches; successful length is validated as 384.
    ('duplicate_of',pa.int64()),('embedding',pa.list_(pa.float32())),
    ('token_count',pa.int64()),('chunk_count',pa.int64()),
])


def export_cache(db, output, identity, stats):
    """Stage all files before publication; caller must hold cache.lock.

    Publication spans multiple renames, not a filesystem transaction. A persistent
    marker makes interrupted publication fail closed until the next successful
    export regenerates the complete snapshot from the SQLite source of truth.
    """
    output = Path(output)
    with TemporaryDirectory(prefix='.export-', dir=output) as directory:
        staged = Path(directory)
        report = _write_export(db, staged, identity, stats)
        marker = output / 'export_in_progress.json'
        json_write(marker, {'state': 'publishing'})
        for name in EXPORT_FILES:
            (staged / name).replace(output / name)
        marker.unlink()
        return report


def _write_export(db, output, identity, stats):
    output=Path(output)
    query='''SELECT o.*,v.embedding,v.token_count,v.chunk_count,f.error_type,f.message
             FROM observations o LEFT JOIN vectors v USING(text_sha256)
             LEFT JOIN failures f USING(text_sha256) ORDER BY source_index'''
    cursor=db.execute(query)
    counts={k:0 for k in ('success','pending','failed','empty','invalid','duplicate')}
    by_month={}; securities=set(); chunks=0
    temporary=output/'filing_embeddings.parquet.tmp'
    with pq.ParquetWriter(temporary, EXPORT_SCHEMA,compression='zstd') as writer:
        while records:=cursor.fetchmany(1024):
            rows=[]
            for idx,doc,permno,day,available,digest,status,reason,duplicate_of,blob,tokens,n_chunks,error,message in records:
                if status=='pending':
                    # LEFT JOIN may match neither table: this text has not been attempted.
                    # observations stores eligibility; vectors/failures store encoding outcomes.
                    if blob is not None: status='success'
                    elif error is not None: status,reason='failed',error+': '+message
                vector = None
                if status == 'success':
                    values = np.frombuffer(blob,dtype='<f4')
                    if values.shape != (384,) or not np.isfinite(values).all():
                        raise ValueError('corrupt cached embedding')
                    vector = values.tolist()
                counts[status]+=1
                month=day[:7] if day else 'unknown'
                by_month.setdefault(month,{k:0 for k in counts})[status]+=1
                if status=='success':securities.add(permno);chunks+=n_chunks
                rows.append(dict(source_index=idx,document_id=doc,permno=permno,
                    filing_date=date.fromisoformat(day) if day else None,
                    available_at_utc=datetime.fromisoformat(available) if available else None,
                    text_sha256=digest,status=status,reason=reason,duplicate_of=duplicate_of,
                    embedding=vector,token_count=tokens if status=='success' else None,
                    chunk_count=n_chunks if status=='success' else None))
            writer.write_table(pa.Table.from_pylist(rows,schema=EXPORT_SCHEMA))
    temporary.replace(output/'filing_embeddings.parquet')
    eligible=counts['success']+counts['pending']+counts['failed']
    report={'source_rows':sum(counts.values()),'counts':counts,'eligible_rows':eligible,
            'source_securities':db.execute('SELECT COUNT(DISTINCT permno) FROM observations').fetchone()[0],
            'source_date_range':list(db.execute('SELECT MIN(filing_date),MAX(filing_date) FROM observations').fetchone()),
            'unique_source_texts':db.execute("SELECT COUNT(DISTINCT text_sha256) FROM observations WHERE status='pending'").fetchone()[0],
            'coverage':counts['success']/eligible if eligible else 0,
            'complete':counts['pending']==counts['failed']==0,
            'fully_usable':counts['pending']==counts['failed']==counts['invalid']==counts['empty']==0,
            'successful_securities':len(securities),'successful_observation_chunks':chunks,
            'unique_vectors':db.execute('SELECT COUNT(*) FROM vectors').fetchone()[0],
            'by_filing_month':dict(sorted(by_month.items())),'last_run':stats,
            'availability_note':'provider filing_date + configured calendar-day delay at UTC midnight; not verified SEC acceptance',
            'scope':'Full-source audit; only success rows carry embeddings. Pending/failed are not no-event months.'}
    json_write(output/'coverage_report.json',report)
    manifest={'identity':identity,'files':{
        name:sha256_file(output/name) for name in ('filing_embeddings.parquet','coverage_report.json')},
        'counts':counts,'complete':report['complete']}
    json_write(output/'manifest.json',manifest)
    text_metadata=text_provenance(identity, sha256_file(output/'manifest.json'))
    json_write(output/'text_metadata.json',text_metadata)
    return report


def text_provenance(identity, manifest_sha256):
    cfg=identity['encoder']
    return {'encoder_name':cfg['model'],'encoder_revision':cfg['revision'],
        'tokenizer_revision':cfg['revision'],'cache_version':VERSION,
        'cache_manifest_sha256':manifest_sha256,
        'preprocessing_version':VERSION,'embedding_dim':384,'frozen':True,
        'month_order':'oldest_to_newest','count_transform':'raw','synthetic':False}


def run_pipeline(source, output, config, *, model_cache_dir, max_documents=None,
                 audit_only=False, retry_failed=False, allow_download=False):
    source,output=Path(source),Path(output)
    if not source.is_file():raise FileNotFoundError(source)
    fields=set(pq.ParquetFile(source).schema_arrow.names)
    if not set(COLUMNS)<=fields:raise ValueError(f'missing source fields: {sorted(set(COLUMNS)-fields)}')
    identity=identity_for(source,config)
    output.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output/'cache.lock'),timeout=0):
        config_path=output/'cache_identity.json'
        if config_path.exists():
            if json.loads(config_path.read_text()) != identity:
                raise ValueError('cache source/model/preprocessing identity mismatch; use a new cache directory')
        else:
            if (output/'cache.sqlite3').exists():raise ValueError('cache database without identity')
            json_write(config_path,identity)
        db=connect(output/'cache.sqlite3')
        stats={'audit_only':audit_only,'max_documents':max_documents,'retry_failed':retry_failed,
               'started_at_utc':datetime.now(timezone.utc).isoformat(),
               'execution_config':asdict(config)}
        try:
            index_source(db,source,config)
            if not audit_only:
                missing=db.execute('''SELECT COUNT(DISTINCT o.text_sha256) FROM observations o
                    LEFT JOIN vectors v USING(text_sha256) LEFT JOIN failures f USING(text_sha256)
                    WHERE o.status='pending' AND v.text_sha256 IS NULL
                    AND (? OR f.text_sha256 IS NULL)''',(retry_failed,)).fetchone()[0]
                if missing:
                    encoder=FrozenMiniLM(config,model_cache_dir,local_files_only=not allow_download)
                    stats.update(encode_pending(db,source,encoder,max_documents=max_documents,retry_failed=retry_failed))
                else:stats.update(new_vectors=0,attempts=0,elapsed_seconds=0)
            return export_cache(db,output,identity,stats)
        except BaseException as exc:
            try:
                if db.execute("SELECT 1 FROM meta WHERE key='indexed'").fetchone():
                    stats.update(interrupted=True,error_type=type(exc).__name__)
                    export_cache(db,output,identity,stats)
            except BaseException as cleanup_error:
                # Preserve even KeyboardInterrupt/SystemExit if cleanup itself fails.
                exc.add_note(f'Cache cleanup export failed: {type(cleanup_error).__name__}: {cleanup_error}')
                LOGGER.error('Cache cleanup export failed; preserving original error', exc_info=True)
            raise
        finally:
            db.close()


def load_verified_cache(output, *, allow_partial=False):
    """Verify exported bytes and vector shapes before E builds event histories.

    Partial mode is an explicit debugging opt-in, never a production default.
    All observation statuses are returned so pending filings cannot become no-event.
    """
    output=Path(output)
    with FileLock(str(output/'cache.lock'),timeout=0):
        if (output/'export_in_progress.json').exists():
            raise ValueError('cache export incomplete; rerun the pipeline to regenerate the snapshot')
        manifest=json.loads((output/'manifest.json').read_text())
        metadata=json.loads((output/'text_metadata.json').read_text())
        if metadata != text_provenance(manifest['identity'], sha256_file(output/'manifest.json')):
            raise ValueError('cache manifest checksum mismatch')
        expected_files={'filing_embeddings.parquet','coverage_report.json'}
        if set(manifest['files']) != expected_files:
            raise ValueError('unexpected cache manifest files')
        for name,digest in manifest['files'].items():
            if sha256_file(output/name)!=digest:
                raise ValueError(f'cache checksum mismatch: {name}')
        report=json.loads((output/'coverage_report.json').read_text())
        if not allow_partial and not report['fully_usable']:
            raise ValueError('cache has pending/failed/empty/invalid observations; not ready for full training')
        table=pq.read_table(output/'filing_embeddings.parquet')
        if table.num_rows!=report['source_rows']:
            raise ValueError('cache row count mismatch')
        for batch in table.select(['status','embedding']).to_batches(max_chunksize=4096):
            for row in batch.to_pylist():
                if row['status']=='success':
                    values=np.asarray(row['embedding'],dtype=np.float32)
                    if values.shape!=(384,) or not np.isfinite(values).all():
                        raise ValueError('cache contains invalid success embedding')
                elif row['embedding'] is not None:
                    raise ValueError('non-success observation contains a misleading embedding')
        return table,metadata,report

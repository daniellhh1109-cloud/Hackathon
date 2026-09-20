"""Verify real local MiniLM freezing/reproducibility and the exported cache."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from src.data.precompute_text_embeddings import (read_config,FrozenMiniLM,source_batches,
    sha256_file,json_write,load_verified_cache)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/text_embeddings.yaml')
    parser.add_argument('--output',default='outputs/text_embeddings/verification.json')
    args=parser.parse_args()
    raw,config=read_config(args.config)
    model=FrozenMiniLM(config,raw['model_cache_dir'])
    table=pq.read_table(raw['source_path'],columns=['text_chars'])
    lengths=np.asarray(table['text_chars'])
    order=np.argsort(lengths)
    selected={int(order[0]),int(order[len(order)//2]),int(order[int(len(order)*.95)])}
    samples=[];idx=0
    for batch in source_batches(raw['source_path']):
        for row in batch.to_pylist():
            if idx in selected:samples.append((idx,row))
            idx+=1
        if len(samples)==len(selected):break
    import hashlib
    def weights_hash():
        digest=hashlib.sha256()
        for name,param in model.model.state_dict().items():
            digest.update(name.encode());digest.update(param.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    before=weights_hash();records=[]
    for idx,row in samples:
        first,tokens,chunks=model.encode(row['text'])
        # Parent code accidentally toggling train is corrected by encode().
        model.model.train()
        second,_,_=model.encode(row['text'])
        np.testing.assert_allclose(first,second,rtol=0,atol=0)
        assert first.shape==(384,) and np.isfinite(first).all()
        assert not model.model.training and all(not p.requires_grad and p.grad is None for p in model.model.parameters())
        records.append({'source_index':idx,'document_id':row['document_id'],
                        'text_chars':len(row['text']),'token_count':tokens,'chunk_count':chunks,
                        'shape':list(first.shape),'repeat_max_difference':float(np.max(np.abs(first-second)))})
    assert weights_hash()==before
    cached,metadata,report=load_verified_cache(raw['cache_dir'],allow_partial=True)
    if not report['fully_usable']:
        try:load_verified_cache(raw['cache_dir'])
        except ValueError:partial_rejected=True
        else:raise AssertionError('production reader accepted incomplete cache')
    else:partial_rejected=None
    snapshot=Path(raw['model_cache_dir'])/('models--'+config.model.replace('/','--'))/'snapshots'/config.revision
    result={'passed':True,'synthetic':False,'encoder_name':config.model,'revision':config.revision,
        'source_sha256':sha256_file(raw['source_path']),'model_files':{p.name:sha256_file(p) for p in snapshot.iterdir() if p.is_file()},
        'weights_unchanged':True,'encoder_frozen':True,'samples':records,
        'cache_rows':cached.num_rows,'cache_counts':report['counts'],
        'partial_cache_rejected_for_training':partial_rejected,'text_metadata':metadata}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);json_write(output,result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

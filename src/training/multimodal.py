"""Member A's cached-embedding training contract; no encoder/model implementation.

E supplies deterministic items with filings[6,K,384], filing_mask[6,K]
and raw filing_counts[6]. Months are oldest to newest. Padding may vary per item.
B/E must verify individual filing timestamps and securities before these items exist.
"""
from functools import partial
import torch
from torch.utils.data import default_collate
from src.training.trainer import fit, load_checkpoint

TEXT_KEYS = {'filings', 'filing_mask', 'filing_counts'}


def validate_text(row):
    if not TEXT_KEYS <= row.keys():
        raise ValueError('missing cached text fields')
    x, mask, counts = (torch.as_tensor(row[k]) for k in ('filings','filing_mask','filing_counts'))
    if x.ndim != 3 or x.shape[0] != 6 or x.shape[2] != 384 or not torch.isfinite(x).all():
        raise ValueError('filings must be finite [6,K,384]')
    if mask.dtype != torch.bool or mask.shape != x.shape[:2]:
        raise ValueError('filing_mask must be bool [6,K]')
    if counts.shape != (6,) or counts.dtype not in (torch.int32, torch.int64) or not torch.equal(counts.long(), mask.sum(-1)):
        raise ValueError('filing_counts must be raw integer counts matching mask')
    return x.float(), mask, counts.long()


def collate_multimodal(rows):
    validated = [validate_text(r) for r in rows]
    batch = default_collate([{k:v for k,v in r.items() if k not in TEXT_KEYS} for r in rows])
    maximum = max(1, max(x.shape[1] for x,_,_ in validated))
    x = torch.zeros(len(rows),6,maximum,384)
    mask = torch.zeros(len(rows),6,maximum,dtype=torch.bool)
    for i,(values,valid,_) in enumerate(validated):
        x[i,:,:values.shape[1]] = values
        mask[i,:,:values.shape[1]] = valid
    batch.update(filings=x, filing_mask=mask, filing_counts=torch.stack([v[2] for v in validated]))
    return batch


def forward_multimodal(model, batch, device):
    # Defense against parent model.train() activating dropout in an attached frozen encoder.
    encoder = getattr(model, 'text_encoder', None)
    if encoder is not None:
        if any(p.requires_grad for p in encoder.parameters()):
            raise ValueError('text_encoder must be frozen')
        encoder.eval()
    return model(quant=batch['quant'].to(device=device,dtype=torch.float32),
                 filings=batch['filings'].to(device=device,dtype=torch.float32),
                 filing_mask=batch['filing_mask'].to(device),
                 filing_counts=batch['filing_counts'].to(device))


def validate_provenance(metadata):
    required = {'encoder_name','encoder_revision','tokenizer_revision','cache_version',
                'cache_manifest_sha256','preprocessing_version','embedding_dim','frozen',
                'month_order','count_transform','synthetic'}
    if not required <= metadata.keys():
        raise ValueError(f'text metadata missing {sorted(required - metadata.keys())}')
    if metadata['embedding_dim'] != 384 or metadata['frozen'] is not True:
        raise ValueError('requires frozen 384-dimensional encoder')
    if metadata['month_order'] != 'oldest_to_newest' or metadata['count_transform'] != 'raw':
        raise ValueError('requires oldest_to_newest months and raw counts')
    if type(metadata['synthetic']) is not bool:
        raise ValueError('synthetic must be explicit boolean')
    for key in required - {'embedding_dim','frozen','synthetic'}:
        if not isinstance(metadata[key],str) or not metadata[key].strip():
            raise ValueError(f'empty text provenance: {key}')
    digest=metadata['cache_manifest_sha256']
    if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('cache manifest requires SHA-256 hex digest')


def _factory(factory):
    model=factory()
    encoder=getattr(model,'text_encoder',None)
    if encoder is not None and any(p.requires_grad for p in encoder.parameters()):
        raise ValueError('text_encoder must be frozen before optimizer construction')
    return model


def fit_multimodal(*, text_metadata, **kwargs):
    validate_provenance(text_metadata)
    coverage={}
    for role in ('train','validation'):
        dataset=kwargs[f'{role}_dataset']
        missing=0
        for row in dataset:
            _,mask,_=validate_text(row)
            missing+=int(not mask.any())
        coverage[role]={'samples':len(dataset),'all_empty_text_samples':missing}
    kwargs['model_factory']=partial(_factory,kwargs['model_factory'])
    return fit(**kwargs, collate_fn=collate_multimodal, batch_forward=forward_multimodal,
               extra_metadata={'modality':'quant_and_cached_filings','text':text_metadata,
                               'text_coverage':coverage})


def load_multimodal_checkpoint(path, model, *, expected_text_metadata, **kwargs):
    # Check provenance BEFORE mutating model parameters.
    validate_provenance(expected_text_metadata)
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('integration_metadata',{}).get('text') != expected_text_metadata:
        raise ValueError('checkpoint text/cache provenance mismatch')
    _factory(lambda:model)
    return load_checkpoint(path,model,**kwargs)

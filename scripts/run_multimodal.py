"""Train a multimodal model from cached filing embeddings, or run a synthetic smoke check."""
import argparse
import hashlib
import importlib
from pathlib import Path
import torch
from torch import nn
from scripts.run import SyntheticDataset
from src.training.trainer import read_config, _json_write
from src.training.multimodal import fit_multimodal, load_multimodal_checkpoint, collate_multimodal, forward_multimodal

class SmokeMultimodal(nn.Module):
    """Trainer probe only: NOT MiniLM, ModernTCN, or C's attention architecture."""
    def __init__(self):
        super().__init__()
        self.text_encoder=nn.Sequential(nn.Linear(2,2), nn.Dropout(.5))
        self.text_encoder.requires_grad_(False)
        self.head=nn.Linear(3,1)
    def forward(self, quant, filings, filing_mask, filing_counts):
        text=(filings[:,:,:,0]*filing_mask).sum((1,2))/filing_mask.sum((1,2)).clamp_min(1)
        return self.head(torch.cat([quant[:,-1,:2],text[:,None]],1))

def smoke_components():
    def samples(n,month,end,seed):
        base=SyntheticDataset(n,month,end,seed)
        rows=[]
        gen=torch.Generator().manual_seed(seed)
        for i in range(n):
            row=base[i]; k=i%4
            values=torch.randn(6,k,384,generator=gen)
            mask=torch.rand(6,k,generator=gen)>.4
            value=(values[:,:,0]*mask).sum()/mask.sum().clamp_min(1)
            row.update(filings=values,filing_mask=mask,filing_counts=mask.sum(-1))
            row['target']=row['target']+.03*value
            rows.append(row)
        return rows
    return dict(model_factory=SmokeMultimodal,
                train_dataset=samples(64,'2018-12','2018-11',10),
                validation_dataset=samples(32,'2020-12','2020-11',11),
                feature_names=[f'synthetic_factor_{i}' for i in range(147)],
                model_metadata={'name':'SmokeMultimodal','synthetic':True,'version':1},
                preprocessing_metadata={'synthetic':True,'return_unit':'decimal'},
                text_metadata={'encoder_name':'synthetic_no_encoder','encoder_revision':'fixture-v1',
                  'tokenizer_revision':'not-applicable-fixture','cache_version':'fixture-v1',
                  'cache_manifest_sha256':hashlib.sha256(b'week3-artificial-fixture-v1').hexdigest(),
                  'preprocessing_version':'fixture-v1','embedding_dim':384,'frozen':True,
                  'month_order':'oldest_to_newest','count_transform':'raw','synthetic':True})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--smoke',action='store_true')
    source.add_argument('--factory')
    parser.add_argument('--config',default='configs/member_a_week3.yaml')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--year',type=int,default=2021)
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.smoke:
        if args.year!=2021: parser.error('smoke fixtures use 2021 split only')
        components=smoke_components()
    else:
        module,name=args.factory.split(':',1)
        components=getattr(importlib.import_module(module),name)(year=args.year)
    config=read_config(args.config)
    model,summary=fit_multimodal(**components,config=config,target_year=args.year,output_dir=args.output_dir)
    restored=components['model_factory']()
    load_multimodal_checkpoint(Path(args.output_dir)/'best.pt',restored,
        expected_text_metadata=components['text_metadata'],expected_feature_names=components['feature_names'],
        expected_model_metadata=components['model_metadata'])
    batch=collate_multimodal([components['validation_dataset'][i] for i in range(min(8,len(components['validation_dataset'])))])
    with torch.no_grad():
        expected=forward_multimodal(model,batch,config.device).cpu()
        actual=forward_multimodal(restored,batch,'cpu')
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    _json_write(Path(args.output_dir)/'reload_check.json',{'passed':True,'synthetic':components['text_metadata']['synthetic'],
        'quant_shape':list(batch['quant'].shape),'filings_shape':list(batch['filings'].shape),
        'prediction_shape':list(actual.shape),'max_absolute_difference':float((actual-expected).abs().max()),
        'summary':summary})
    print('Independent multimodal checkpoint reload verified.')
if __name__=='__main__': main()

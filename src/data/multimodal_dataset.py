"""Point-in-time six-month event windows aligned to existing quant samples."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.data.precompute_text_embeddings import load_verified_cache
from src.data.quant_dataset import QuantDataset
from src.training.multimodal_evaluation import _event_frame, event_coverage, KEYS
from src.inference.predict_month import canonical_month


class EventStore:
    """One verified cache snapshot shared by training and validation datasets.

    Partial global caches are readable, but every requested sample must have
    complete eligible events. No document truncation or success-based sampling.
    """
    def __init__(self, cache_dir):
        table,self.metadata,self.report=load_verified_cache(cache_dir,allow_partial=True)
        columns=['source_index','permno','filing_date','available_at_utc','status']
        self.events=_event_frame(table.select(columns).to_pandas())
        self.vectors={}
        for batch in table.select(['source_index','status','embedding']).to_batches(max_chunksize=4096):
            for row in batch.to_pylist():
                if row['status']=='success':
                    self.vectors[row['source_index']]=np.asarray(row['embedding'],dtype=np.float32)
        # One index per security/month, not one copy of events per sample.
        self.events=self.events.sort_values(['_available','source_index']).reset_index(drop=True)
        self.groups=self.events.groupby(['permno','_filing_month'],sort=False).indices

    def evidence(self, permno, target_month):
        target=canonical_month(target_month)
        cutoff=target.start_time.tz_localize('UTC')
        groups=[]
        for month in pd.period_range(target-6,target-1,freq='M'):
            positions=self.groups.get((int(permno),month))
            if positions is not None:
                group=self.events.iloc[positions]
                groups.append(group[(group['_available']<cutoff)&~group.status.eq('duplicate')])
        result=pd.concat(groups,ignore_index=True) if groups else self.events.iloc[:0].copy()
        if not result.status.eq('success').all():
            raise ValueError(f'incomplete text window: permno={permno}, target_month={target_month}')
        return result

    def tensors(self, permno, target_month):
        selected=self.evidence(permno,target_month)
        target=canonical_month(target_month)
        months=[selected[selected['_filing_month']==target-6+i] for i in range(6)]
        counts=torch.tensor([len(group) for group in months],dtype=torch.int64)
        maximum=int(counts.max())
        values=torch.zeros(6,maximum,384,dtype=torch.float32)
        mask=torch.zeros(6,maximum,dtype=torch.bool)
        for i,group in enumerate(months):
            for j,source in enumerate(group.source_index):
                values[i,j]=torch.from_numpy(self.vectors[source])
                mask[i,j]=True
        return dict(filings=values,filing_mask=mask,filing_counts=counts)


class MultimodalDataset(Dataset):
    def __init__(self,store_dir,event_store,*,year,partition,supervised=None,month=None):
        self.quant_dataset=QuantDataset(store_dir,year=year,partition=partition,supervised=supervised)
        self.event_store=event_store
        self.indices=np.arange(len(self.quant_dataset))
        if month is not None:
            canonical_month(month)
            self.indices=self.indices[self.quant_dataset.samples.target_month.eq(month).to_numpy()]
        self.samples=self.quant_dataset.samples.iloc[self.indices].reset_index(drop=True)
        if self.samples.empty:raise ValueError('no eligible samples for requested split/month')
        self.coverage=event_coverage(self.samples[KEYS],event_store.events)
        incomplete=int((~self.coverage.text_complete).sum())
        if incomplete:
            raise ValueError(f'incomplete text coverage: {incomplete} of {len(self.samples)} sample windows; finish encoding, do not drop samples')
        self.feature_names=self.quant_dataset.feature_names
        self.metadata=self.quant_dataset.metadata
        self.text_metadata=event_store.metadata

    def __len__(self):return len(self.indices)

    def __getitem__(self,index):
        row=self.quant_dataset[int(self.indices[index])]
        row.update(self.event_store.tensors(row['permno'],row['target_month']))
        return row

    def evidence(self,index):
        row=self.samples.iloc[index]
        return self.event_store.evidence(int(row.permno),row.target_month).drop(columns=['_available','_filing_month'])

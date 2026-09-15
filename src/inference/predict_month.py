"""Monthly, label-free inference from a completed annual quant checkpoint.

Rebuilds the model from checkpoint metadata, never from mutable model.yaml.
No realized-target file is opened by this module or its validator.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from src.data.quant_dataset import QuantDataset, make_loader, sha256, write_json
from src.data.splits import annual_split
from src.models.modern_tcn import QuantModelConfig
from src.models.quant_model import QuantRegressor, model_metadata
from src.training.trainer import load_checkpoint

CORE_COLUMNS = ['target_month','permno','predicted_excess_return','rank',
                'quant_end_month','model_year','checkpoint_sha256']


def canonical_month(value):
    if not isinstance(value,str) or len(value)!=7:
        raise ValueError('month must be YYYY-MM')
    try:
        parsed=pd.Period(value,freq='M')
    except (ValueError,TypeError) as exc:
        raise ValueError('month must be YYYY-MM') from exc
    if str(parsed)!=value:
        raise ValueError('month must be YYYY-MM')
    return parsed


def validate_predictions(table, *, month, expected_permnos, model_year, checkpoint_sha256):
    """Strict schema, exact universe and deterministic descending rank validation."""
    period=canonical_month(month)
    if table.empty or list(table.columns)!=CORE_COLUMNS:
        raise ValueError('empty predictions or unexpected columns')
    if not table.target_month.eq(month).all() or period.year != model_year:
        raise ValueError('target month/model year mismatch')
    if not table.model_year.eq(model_year).all() or not table.checkpoint_sha256.eq(checkpoint_sha256).all():
        raise ValueError('model provenance mismatch')
    if not table.quant_end_month.eq(str(period-1)).all():
        raise ValueError('information cutoff must be previous calendar month')
    ids=pd.to_numeric(table.permno,errors='raise')
    if not np.isfinite(ids).all() or (ids<=0).any() or (ids%1!=0).any() or ids.duplicated().any():
        raise ValueError('invalid or duplicate security IDs')
    expected=list(expected_permnos)
    if len(expected)!=len(set(expected)) or set(ids)!=set(expected):
        raise ValueError('predicted securities differ from expected universe')
    prediction=pd.to_numeric(table.predicted_excess_return,errors='raise')
    if not np.isfinite(prediction).all():
        raise ValueError('non-finite predictions')
    ordered=table.sort_values(['predicted_excess_return','permno'],ascending=[False,True]).reset_index(drop=True)
    if not table.reset_index(drop=True).equals(ordered) or not np.array_equal(table['rank'].to_numpy(),np.arange(1,len(table)+1)):
        raise ValueError('incorrect descending ranking; ties use ascending permno')
    return {'target_month':month,'rows':len(table),'securities':int(ids.nunique()),
            'prediction_min':float(prediction.min()),'prediction_max':float(prediction.max()),
            'prediction_mean':float(prediction.mean()),'finite_predictions':True,'exact_universe':True}


class MonthlyPredictor:
    def __init__(self, store_dir, checkpoint_path, *, year, batch_size=512):
        if type(batch_size) is not int or batch_size<1:
            raise ValueError('batch_size must be positive')
        self.year,self.batch_size=year,batch_size
        self.store_dir=Path(store_dir)
        self.checkpoint_path=Path(checkpoint_path)
        self.dataset=QuantDataset(store_dir,year=year,partition='test',supervised=False)
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        if checkpoint['target_year']!=year:
            raise ValueError('cannot use another year checkpoint for historical predictions')
        if checkpoint['preprocessing_metadata']!=self.dataset.metadata:
            raise ValueError('checkpoint preprocessing/data metadata mismatch')
        split=annual_split(year)
        expected_bounds={part:[getattr(split,f'{part}_{edge}').strftime('%Y-%m') for edge in ['start','end']]
                         for part in ['train','validation','test']}
        if checkpoint['annual_bounds']!=expected_bounds:
            raise ValueError('checkpoint annual split mismatch')
        for part in ['train','validation']:
            observed=checkpoint['data_audit'][part]['target_month_range']
            low,high=expected_bounds[part]
            if len(observed)!=2 or not low<=observed[0]<=observed[1]<=high:
                raise ValueError('checkpoint used out-of-scope training/validation months')
        summary=json.loads(self.checkpoint_path.with_name('summary.json').read_text())
        if summary['best_epoch']!=checkpoint['epoch'] or summary['best_validation_loss']!=checkpoint['validation_loss']:
            raise ValueError('checkpoint is not the completed training selection')
        config=QuantModelConfig(**checkpoint['model_metadata']['config'])
        self.model=QuantRegressor(config)
        load_checkpoint(checkpoint_path,self.model,expected_feature_names=self.dataset.feature_names,
                        expected_model_metadata=model_metadata(config))
        self.checkpoint_sha256=sha256(checkpoint_path)
        self.checkpoint=checkpoint
        # Only calendar/identity columns: never read posterior names or realized labels.
        self.context=pd.read_parquet(self.store_dir/'raw_context.parquet',columns=['permno','date','eom','target_month'])
        self.windows=pd.read_parquet(self.store_dir/'windows.parquet')
        if len(self.context)!=len(self.dataset.quant) or len(self.windows)!=len(self.context):
            raise ValueError('store row-count mismatch')

    def predict(self, month):
        period=canonical_month(month)
        bounds=self.checkpoint['annual_bounds']['test']
        if not bounds[0]<=month<=bounds[1]:
            raise ValueError('month outside annual test interval')
        ids=self.dataset.samples.index[self.dataset.samples.target_month.eq(month)].to_numpy()
        if not len(ids): raise ValueError(f'no eligible securities for {month}')
        selected=self.dataset.samples.iloc[ids]
        rows=selected.row_index.to_numpy()
        if (rows<11).any() or (rows>=len(self.context)).any():
            raise ValueError('invalid window row index')
        positions=rows[:,None]+np.arange(-11,1)[None,:]
        ctx=self.context
        month_ord=pd.to_datetime(ctx.eom).dt.to_period('M').astype('int64').to_numpy()
        expected=np.arange(period.ordinal-12,period.ordinal)
        if not np.array_equal(month_ord[positions],np.broadcast_to(expected,positions.shape)):
            raise ValueError('nonconsecutive/future feature months')
        permnos=selected.permno.to_numpy()
        if not np.array_equal(ctx.permno.to_numpy()[positions],np.broadcast_to(permnos[:,None],positions.shape)):
            raise ValueError('mixed security feature window')
        observed=pd.to_datetime(ctx.date).to_numpy()[positions]
        ends=pd.to_datetime(ctx.eom).to_numpy()[positions]
        if (observed>ends).any() or not pd.to_datetime(ctx.eom.iloc[positions.ravel()]).dt.is_month_end.all():
            raise ValueError('source observation date or month-end invalid')
        if not np.array_equal(pd.to_datetime(observed.ravel()).to_period('M').asi8.reshape(positions.shape),month_ord[positions]):
            raise ValueError('source dates outside feature month')
        values=[]
        with torch.inference_mode():
            for batch in make_loader(Subset(self.dataset,ids.tolist()),batch_size=self.batch_size):
                # Test Dataset has no target key; raise if a future edit violates this interface.
                if 'target' in batch: raise ValueError('realized target exposed to inference')
                prediction=self.model(batch['quant'])
                if prediction.shape!=(len(batch['quant']),1): raise ValueError('bad prediction shape')
                values.append(prediction[:,0].cpu().numpy())
        table=pd.DataFrame({'target_month':month,'permno':permnos,'predicted_excess_return':np.concatenate(values),
                            'quant_end_month':str(period-1),'model_year':self.year,'checkpoint_sha256':self.checkpoint_sha256})
        table=table.sort_values(['predicted_excess_return','permno'],ascending=[False,True]).reset_index(drop=True)
        table['rank']=np.arange(1,len(table)+1)
        table=table[CORE_COLUMNS]
        report=validate_predictions(table,month=month,expected_permnos=permnos,model_year=self.year,
                                    checkpoint_sha256=self.checkpoint_sha256)
        source=self.windows[self.windows.target_month.eq(month)]
        report.update({'raw_candidate_rows':len(source),
                       'excluded_insufficient_history':int(source.window_status.eq('insufficient_history').sum()),
                       'excluded_calendar_gap':int(source.window_status.eq('calendar_gap').sum()),
                       'label_based_exclusions':0,'sample_input_shape':[12,147]})
        if report['rows']+report['excluded_insufficient_history']+report['excluded_calendar_gap']!=len(source):
            raise ValueError('monthly eligibility counts do not reconcile')
        return table,report

    def export(self, output_dir, *, month=None):
        output_dir=Path(output_dir)
        if output_dir.exists(): raise FileExistsError(output_dir)
        bounds=self.checkpoint['annual_bounds']['test']
        months=[month] if month else pd.period_range(*bounds,freq='M').astype(str).tolist()
        # Validate requested month before creating an output directory.
        for value in months:
            canonical_month(value)
            if not bounds[0]<=value<=bounds[1]: raise ValueError('month outside annual test interval')
        output_dir.mkdir(parents=True)
        provenance={'schema_version':1,'year':self.year,'requested_months':months,'return_unit':'decimal',
                    'checkpoint':str(self.checkpoint_path.resolve()),'checkpoint_sha256':self.checkpoint_sha256,
                    'selected_epoch':self.checkpoint['epoch'],'model_metadata':self.checkpoint['model_metadata'],
                    'preprocessing_metadata':self.dataset.metadata,'batch_size':self.batch_size,
                    'cpu_threads':torch.get_num_threads(),'torch_version':str(torch.__version__),
                    'input_artifacts_sha256':{name:sha256(self.store_dir/name) for name in
                        ['metadata.json','quant.npy','windows.parquet','raw_context.parquet']},
                    'labels_accessed':False,'field_note':'rank=1 is highest prediction; ties ordered by permno; no name/ticker joins'}
        write_json(output_dir/'provenance.json',provenance)
        reports=[];files={}
        for value in months:
            table,report=self.predict(value)
            parquet=output_dir/f'{value}.parquet';csv=output_dir/f'{value}.csv'
            table.to_parquet(parquet,index=False)
            table.to_csv(csv,index=False,float_format='%.17g')
            # Read-back checks guard serialization, column order and lost rows.
            for path in [parquet,csv]:
                restored=pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)
                validate_predictions(restored,month=value,expected_permnos=table.permno.tolist(),
                                     model_year=self.year,checkpoint_sha256=self.checkpoint_sha256)
                if not np.allclose(restored.predicted_excess_return,table.predicted_excess_return,rtol=0,atol=1e-15):
                    raise ValueError('serialized predictions changed')
                files[path.name]=sha256(path)
            reports.append(report)
            print(f'{value}: {len(table)} predictions, checked CSV/Parquet',flush=True)
        pd.DataFrame(reports).to_csv(output_dir/'monthly_audit.csv',index=False)
        result={'status':'complete','year':self.year,'months':len(months),'predictions':sum(r['rows'] for r in reports),
                'checkpoint_sha256':self.checkpoint_sha256,'files_sha256':files,'labels_accessed':False}
        # Completion marker only appears after all requested files have passed checks.
        write_json(output_dir/'manifest.json',result)
        return result

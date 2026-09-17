"""Production dataset/model factory for run_multimodal; never uses fixtures."""
from functools import partial
from pathlib import Path
import yaml
from src.data.multimodal_dataset import EventStore,MultimodalDataset
from src.models.multimodal_model import MultimodalRegressor,read_multimodal_config,model_metadata


def build_components(*,year=2021,config_path='configs/multimodal_data.yaml'):
    raw=yaml.safe_load(Path(config_path).read_text())
    events=EventStore(raw['cache_dir'])
    config=read_multimodal_config(raw['model_config'])
    train=MultimodalDataset(raw['store_dir'],events,year=year,partition='train')
    validation=MultimodalDataset(raw['store_dir'],events,year=year,partition='validation')
    if train.feature_names!=validation.feature_names:raise ValueError('factor order mismatch')
    return dict(model_factory=partial(MultimodalRegressor,config),train_dataset=train,validation_dataset=validation,
                feature_names=train.feature_names,model_metadata=model_metadata(config),
                preprocessing_metadata=train.metadata,text_metadata=events.metadata)

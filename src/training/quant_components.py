"""Assemble full-universe datasets and the shared quant model for the training API."""
from functools import partial
from pathlib import Path
import yaml
from src.data.quant_dataset import QuantDataset
from src.models.quant_model import QuantRegressor, read_model_config, model_metadata


def build_components(dataset_config='configs/datasets.yaml', model_config='configs/model.yaml', year=None):
    data = yaml.safe_load(Path(dataset_config).read_text())
    selected_year = data['year'] if year is None else year
    config = read_model_config(model_config)
    train = QuantDataset(data['store_dir'], year=selected_year, partition='train')
    validation = QuantDataset(data['store_dir'], year=selected_year, partition='validation')
    if train.feature_names != validation.feature_names:
        raise ValueError('train/validation factor order mismatch')
    return {'model_factory': partial(QuantRegressor,config), 'train_dataset': train,
            'validation_dataset': validation, 'feature_names': train.feature_names,
            'model_metadata': model_metadata(config), 'preprocessing_metadata': train.metadata}

"""Member A: quant-only training, validation selection and portable checkpoints.

Dataset items: quant[12,147], target scalar, target_month/quant_end_month
(YYYY-MM), permno. Dataset construction and ModernTCN belong to B/C.
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml
from src.data.splits import annual_split


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    lr: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 512
    max_epochs: int = 50
    early_stopping_patience: int = 5
    min_delta: float = 0.0
    huber_delta: float = 1.0
    device: str = 'cpu'

    def __post_init__(self):
        for name in ('seed', 'batch_size', 'max_epochs', 'early_stopping_patience'):
            value = getattr(self, name)
            minimum = 0 if name == 'seed' else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        for name in ('lr', 'weight_decay', 'min_delta', 'huber_delta'):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
            if value < 0 or (name in ('lr', 'huber_delta') and value == 0):
                raise ValueError(f'invalid {name}')
        if self.device not in ('cpu', 'cuda', 'mps'):
            raise ValueError('device must be cpu, cuda or mps')


def read_config(path):
    with open(path, encoding='utf-8') as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or set(raw) != {'training'}:
        raise ValueError('member A config must have exactly one training section')
    return TrainingConfig(**raw['training'])


def seed_everything(seed):
    """Call before model construction; CPU is the reference reproducibility target."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def _month(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', value):
        raise ValueError('months must be YYYY-MM strings')
    year, month = map(int, value.split('-'))
    return year * 12 + month - 1


def annual_bounds(year):
    """Use the same target-month boundaries as data construction and inference."""
    split = annual_split(year)
    return {part: [getattr(split, f'{part}_{edge}').strftime('%Y-%m')
                   for edge in ('start', 'end')]
            for part in ('train', 'validation', 'test')}


def _json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _audit_dataset(dataset, role, bounds):
    if len(dataset) == 0:
        raise ValueError(f'{role} dataset is empty')
    seen, months, end_months = set(), [], []
    required = {'quant', 'target', 'target_month', 'quant_end_month', 'permno'}
    low, high = map(_month, bounds)
    for index in range(len(dataset)):
        row = dataset[index]
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f'{role} item must contain {sorted(required)}')
        quant, target = torch.as_tensor(row['quant']), torch.as_tensor(row['target'])
        if quant.shape != (12, 147) or not torch.isfinite(quant).all():
            raise ValueError(f'{role}: quant must be finite [12,147]')
        if target.ndim != 0 or not torch.isfinite(target):
            raise ValueError(f'{role}: target must be a finite scalar')
        month, end = _month(row['target_month']), _month(row['quant_end_month'])
        if month != end + 1:
            raise ValueError('target must be exactly the next calendar month')
        if not low <= month <= high:
            raise ValueError(f'{role}: target month outside annual split')
        identifier = row['permno']
        if isinstance(identifier, bool) or not isinstance(identifier, (int, np.integer)) or identifier <= 0:
            raise ValueError('permno must be a positive integer')
        key = (int(identifier), month)
        if key in seen:
            raise ValueError(f'{role}: duplicate security-target month')
        seen.add(key)
        months.append(row['target_month'])
        end_months.append(row['quant_end_month'])
    return {'n_samples': len(dataset), 'target_month_range': [min(months), max(months)],
            'quant_end_month_range': [min(end_months), max(end_months)],
            'sample_quant_shape': [12, 147]}


def _vector(value, batch_size, name):
    if value.shape == (batch_size, 1):
        value = value[:, 0]
    if value.shape != (batch_size,):
        raise ValueError(f'{name} must have shape [B] or [B,1]; got {tuple(value.shape)}')
    if not torch.isfinite(value).all():
        raise ValueError(f'non-finite {name}')
    return value


def _run_epoch(model, loader, criterion, device, optimizer=None, batch_forward=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            quant = batch['quant'].to(device=device, dtype=torch.float32)
            target = batch['target'].to(device=device, dtype=torch.float32)
            if quant.ndim != 3 or tuple(quant.shape[1:]) != (12, 147) or not torch.isfinite(quant).all():
                raise ValueError('batch quant must be finite [B,12,147]')
            size = quant.shape[0]
            target = _vector(target, size, 'target')
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = _vector(model(quant) if batch_forward is None else batch_forward(model, batch, device), size, 'prediction')
            loss = criterion(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError('non-finite loss')
            if training:
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise ValueError('non-finite gradient')
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise ValueError('non-finite model parameter')
            total += loss.item() * size
            count += size
    if count == 0:
        raise ValueError('empty loader')
    return total / count


def _validated_metadata(name, value):
    """Normalize JSON metadata and identify the failing argument on invalid input."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'{name} must be JSON-serializable with finite numbers '
            f'(no NaN/Infinity) and no circular references: {exc}'
        ) from exc


def fit(model_factory: Callable[[], nn.Module], train_dataset: Dataset,
        validation_dataset: Dataset, *, config: TrainingConfig, target_year: int,
        output_dir, feature_names: list[str], model_metadata: dict,
        preprocessing_metadata: dict, collate_fn=None, batch_forward=None,
        extra_metadata: dict | None = None):
    """No test dataset accepted. Fresh output directory required; no resume implied.

    model_factory must return a NEW model (seed is set before calling it).
    Datasets must be deterministic/re-iterable and contain only model inputs +
    training labels and audit identifiers. B owns whitelist and window audits.
    """
    if len(feature_names) != 147 or len(set(feature_names)) != 147 or any(not isinstance(n, str) for n in feature_names):
        raise ValueError('feature_names must be 147 unique names in model input order')
    forbidden = {'ret_exc_lead1m', 'ret', 'ret_exc', 'permno', 'target', 'target_month'}
    if forbidden.intersection(n.lower() for n in feature_names):
        raise ValueError('label/identifier in feature_names')
    if not model_metadata or not preprocessing_metadata:
        raise ValueError('model and preprocessing metadata are required')
    # Validate serializability before creating any artifacts.
    model_metadata = _validated_metadata('model_metadata', model_metadata)
    preprocessing_metadata = _validated_metadata('preprocessing_metadata', preprocessing_metadata)
    extra_metadata = _validated_metadata('extra_metadata', extra_metadata if extra_metadata is not None else {})
    bounds = annual_bounds(target_year)
    audit = {role: _audit_dataset(dataset, role, bounds[role]) for role, dataset in
             [('train', train_dataset), ('validation', validation_dataset)]}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(config.seed)
    device = torch.device(config.device)
    model = model_factory().to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('model has no trainable parameters')
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.HuberLoss(delta=config.huber_delta)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              generator=generator, num_workers=0, drop_last=False, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size,
                                   shuffle=False, num_workers=0, drop_last=False, collate_fn=collate_fn)
    metadata = {'format_version': 1, 'training': asdict(config), 'target_year': target_year,
                'annual_bounds': bounds, 'data_audit': audit, 'feature_names': feature_names,
                'model_metadata': model_metadata, 'preprocessing_metadata': preprocessing_metadata,
                'torch_version': str(torch.__version__), 'selection_metric': 'validation_huber_loss',
                'integration_metadata': extra_metadata,
                'trainable_parameter_names': [n for n,p in model.named_parameters() if p.requires_grad],
                'frozen_parameter_names': [n for n,p in model.named_parameters() if not p.requires_grad]}
    _json_write(output_dir / 'run.json', metadata)
    print(json.dumps({'data_audit': audit, 'first_train_batch_quant_shape': [min(config.batch_size, len(train_dataset)), 12, 147]}, ensure_ascii=False), flush=True)
    best, best_epoch, patience_best, stale = math.inf, 0, math.inf, 0
    history = []
    with (output_dir / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'validation_loss', 'best_epoch', 'stale_epochs'])
        writer.writeheader()
        for epoch in range(1, config.max_epochs + 1):
            epoch_kwargs = {} if batch_forward is None else {'batch_forward': batch_forward}
            train_loss = _run_epoch(model, train_loader, criterion, device, optimizer, **epoch_kwargs)
            validation_loss = _run_epoch(model, validation_loader, criterion, device, **epoch_kwargs)
            # Always save the true minimum; min_delta affects patience only.
            if validation_loss < best:
                best, best_epoch = validation_loss, epoch
                checkpoint = {**metadata, 'epoch': epoch, 'validation_loss': best,
                              'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
                temporary = output_dir / 'best.pt.tmp'
                torch.save(checkpoint, temporary)
                temporary.replace(output_dir / 'best.pt')
            if validation_loss < patience_best - config.min_delta:
                patience_best, stale = validation_loss, 0
            else:
                stale += 1
            row = {'epoch': epoch, 'train_loss': train_loss, 'validation_loss': validation_loss,
                   'best_epoch': best_epoch, 'stale_epochs': stale}
            history.append(row)
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row), flush=True)
            if stale >= config.early_stopping_patience:
                break
    load_checkpoint(output_dir / 'best.pt', model, expected_feature_names=feature_names,
                    expected_model_metadata=model_metadata)
    summary = {'best_epoch': best_epoch, 'best_validation_loss': best, 'epochs_run': len(history),
               'stop_reason': 'early_stopping' if stale >= config.early_stopping_patience else 'max_epochs',
               'checkpoint': str(output_dir / 'best.pt')}
    _json_write(output_dir / 'summary.json', summary)
    return model, summary


def load_checkpoint(path, model: nn.Module, *, expected_feature_names: list[str],
                    expected_model_metadata: dict):
    """Load tensor/primitive checkpoint, verify input order and architecture, enter eval."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1:
        raise ValueError('unsupported checkpoint format')
    if checkpoint['feature_names'] != expected_feature_names:
        raise ValueError('checkpoint feature order mismatch')
    if checkpoint['model_metadata'] != expected_model_metadata:
        raise ValueError('checkpoint architecture metadata mismatch')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return checkpoint

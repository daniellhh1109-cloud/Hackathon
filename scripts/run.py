"""Run member A's synthetic training acceptance, or a B/C integration factory."""
import argparse
import importlib
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from src.training.trainer import fit, load_checkpoint, read_config, _json_write


class SyntheticDataset(Dataset):
    """Artificial samples, not competition data or a performance benchmark."""
    def __init__(self, count, target_month, quant_end_month, seed):
        generator = torch.Generator().manual_seed(seed)
        self.quant = torch.randn(count, 12, 147, generator=generator)
        self.target = 0.04 * self.quant[:, -1, 0] - 0.02 * self.quant[:, -1, 1]
        self.target_month, self.quant_end_month = target_month, quant_end_month

    def __len__(self):
        return len(self.quant)

    def __getitem__(self, index):
        return {'quant': self.quant[index], 'target': self.target[index],
                'target_month': self.target_month, 'quant_end_month': self.quant_end_month,
                'permno': index + 1}


class SmokeRegressor(nn.Module):
    """Two-feature linear model for trainer acceptance, explicitly NOT ModernTCN."""
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 1)

    def forward(self, quant):
        return self.head(quant[:, -1, :2])


def smoke_components():
    return {
        'model_factory': SmokeRegressor,
        'train_dataset': SyntheticDataset(64, '2018-12', '2018-11', 10),
        'validation_dataset': SyntheticDataset(32, '2020-12', '2020-11', 11),
        'feature_names': [f'synthetic_factor_{i}' for i in range(147)],
        'model_metadata': {'name': 'SmokeRegressor', 'input_features_used': 2, 'synthetic': True},
        'preprocessing_metadata': {'name': 'synthetic_normal_inputs', 'return_unit': 'decimal', 'synthetic': True},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/member_a_week2.yaml')
    parser.add_argument('--output-dir', required=True, help='New directory; existing runs are never overwritten')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--smoke', action='store_true', help='Artificial data and a tiny linear model only')
    source.add_argument('--factory', help='Python module:function returning the B/C component dictionary')
    parser.add_argument('--year', type=int, default=2021)
    args = parser.parse_args()
    if args.smoke and args.year != 2021:
        parser.error('synthetic acceptance uses the 2021 split only')
    if args.smoke:
        components = smoke_components()
    else:
        module, name = args.factory.split(':', 1)
        components = getattr(importlib.import_module(module), name)()
    config = read_config(args.config)
    model, summary = fit(**components, config=config, target_year=args.year, output_dir=args.output_dir)
    # Independent reload verifies exactly the artifact E will consume.
    restored = components['model_factory']()
    load_checkpoint(Path(args.output_dir) / 'best.pt', restored,
                    expected_feature_names=components['feature_names'],
                    expected_model_metadata=components['model_metadata'])
    quant = torch.as_tensor(components['validation_dataset'][0]['quant']).float().unsqueeze(0)
    with torch.no_grad():
        expected = model(quant.to(config.device)).cpu()
        actual = restored(quant)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    _json_write(Path(args.output_dir) / 'reload_check.json',
                {'passed': True, 'input_shape': list(quant.shape), 'output_shape': list(actual.shape),
                 'synthetic': args.smoke, 'summary': summary})
    print('Independent checkpoint reload verified.', flush=True)


if __name__ == '__main__':
    main()

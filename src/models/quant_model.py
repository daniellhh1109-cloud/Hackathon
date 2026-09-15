"""One shared quant-only regressor with a detachable encoder for week 3."""
from dataclasses import asdict
from pathlib import Path
import yaml
from torch import nn
from src.models.modern_tcn import QuantModelConfig, ModernTCNEncoder


def read_model_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or set(raw) != {'model'}:
        raise ValueError('model config must contain exactly a model section')
    return QuantModelConfig(**raw['model'])


def model_metadata(config):
    return {'name': 'QuantRegressor', 'architecture_version': 'moderntcn_latent_groups_v1',
            'config': asdict(config), 'adaptation': 'projected latent groups; LayerNorm; single temporal branch'}


class QuantRegressor(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or QuantModelConfig()
        self.encoder = ModernTCNEncoder(self.config)
        self.head = nn.Sequential(nn.Linear(self.config.d_model,self.config.head_hidden),
                                  nn.GELU(), nn.Linear(self.config.head_hidden,1))

    def forward(self, quant):
        return self.head(self.encoder(quant))

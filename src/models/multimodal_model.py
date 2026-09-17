"""Shared ModernTCN + event memory + gated residual next-month return model."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import torch
from torch import nn
import yaml
from src.models.modern_tcn import ModernTCNEncoder, QuantModelConfig
from src.models.text_attention import EventMemory


@dataclass(frozen=True)
class MultimodalConfig:
    quant: QuantModelConfig = field(default_factory=QuantModelConfig)
    gate_mode: str = 'scalar'

    def __post_init__(self):
        if not isinstance(self.quant, QuantModelConfig):
            raise ValueError('quant must be QuantModelConfig')
        if self.gate_mode not in ('scalar', 'vector'):
            raise ValueError('gate_mode must be scalar or vector')


def read_multimodal_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or set(raw) != {'model'} or not isinstance(raw['model'], dict):
        raise ValueError('configuration requires one model mapping')
    config = dict(raw['model'])
    quant = config.pop('quant', {})
    if not isinstance(quant, dict):
        raise ValueError('model.quant must be a mapping')
    return MultimodalConfig(quant=QuantModelConfig(**quant), **config)


def model_metadata(config):
    return {'name': 'MultimodalRegressor', 'architecture_version': 'moderntcn_event_memory_v1',
            'config': asdict(config), 'text_input': 'cached_frozen_minilm_384',
            'month_order': 'oldest_to_newest', 'ages': [5,4,3,2,1,0],
            'gate_count': 'log1p_sum_of_raw_counts_over_six_months',
            'empty_text_policy': 'exact_zero_residual_correction',
            'quant_adaptation': 'projected latent groups; LayerNorm; single temporal branch'}


class MultimodalRegressor(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or MultimodalConfig()
        self.quant_encoder = ModernTCNEncoder(self.config.quant)
        self.event_memory = EventMemory()
        width = 1 if self.config.gate_mode == 'scalar' else 128
        self.gate = nn.Sequential(nn.Linear(258,64), nn.GELU(), nn.Linear(64,width), nn.Sigmoid())
        self.text_residual = nn.Linear(128,128,bias=False)
        self.fusion_norm = nn.LayerNorm(128)
        self.head = nn.Sequential(nn.Linear(128,64), nn.GELU(), nn.Linear(64,1))

    def forward(self, quant, filings, filing_mask, filing_counts, *, return_diagnostics=False):
        if filings.ndim != 4 or quant.ndim != 3 or filings.shape[0] != quant.shape[0]:
            raise ValueError('quant and filings require matching batch dimensions')
        if quant.device != filings.device:
            raise ValueError('quant and filings must be on the same device')
        h_quant = self.quant_encoder(quant)
        h_text, details = self.event_memory(filings,filing_mask,filing_counts,return_diagnostics=True)
        has_events = details['has_events'].unsqueeze(-1)
        # Input counts are raw; log1p is applied exactly once, over the full six-month memory.
        intensity = torch.log1p(filing_counts.sum(-1).to(dtype=h_quant.dtype)).unsqueeze(-1)
        gate_input = torch.cat((h_quant,h_text,has_events.to(h_quant.dtype),intensity),dim=-1)
        gate = self.gate(gate_input)
        correction = (gate*self.text_residual(h_text)).masked_fill(~has_events,0)
        fused = self.fusion_norm(h_quant+correction)
        prediction = self.head(fused)
        if return_diagnostics:
            return prediction, {**details,'h_quant':h_quant,'h_text':h_text,'gate':gate,
                                'text_correction':correction,'event_intensity':intensity}
        return prediction

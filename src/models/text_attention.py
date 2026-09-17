"""Trainable filing attention and six-month event memory over cached embeddings."""
import torch
from torch import nn


def masked_softmax(scores, mask):
    """Zero probability on padding, including an entirely empty group."""
    if scores.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError('attention requires matching scores and boolean mask')
    if scores.shape[-1] == 0:
        return torch.zeros_like(scores)
    masked = scores.masked_fill(~mask, float('-inf'))
    # Never evaluate softmax on an all-negative-infinity row (NaN backward too).
    safe = torch.where(mask.any(-1, keepdim=True), masked, torch.zeros_like(masked))
    return torch.softmax(safe, dim=-1).masked_fill(~mask, 0)


class EventMemory(nn.Module):
    """[B,6,K,384] -> [B,128]; oldest-to-newest months, ages 5..0.

    Additive attention uses learned queries implemented as bias-free scalar
    scoring layers. There is no trainable MiniLM here: embeddings are inputs.
    """
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(384, 128)
        self.filing_key = nn.Linear(128, 128)
        self.filing_score = nn.Linear(128, 1, bias=False)
        self.age_embedding = nn.Embedding(6, 128)
        self.temporal_key = nn.Linear(128, 128)
        self.temporal_score = nn.Linear(128, 1, bias=False)
        self.register_buffer('ages', torch.arange(5, -1, -1))

    def forward(self, filings, filing_mask, filing_counts, *, return_diagnostics=False):
        if filings.ndim != 4 or filings.shape[0] < 1 or filings.shape[1] != 6 or filings.shape[-1] != 384:
            raise ValueError('filings must have shape [B,6,K,384] with B>0')
        if not filings.is_floating_point() or not torch.isfinite(filings).all():
            raise ValueError('filings must be finite floating point')
        if filing_mask.dtype != torch.bool or filing_mask.shape != filings.shape[:-1]:
            raise ValueError('filing_mask must be boolean [B,6,K]')
        if filing_counts.dtype not in (torch.int32, torch.int64) or filing_counts.shape != filings.shape[:2]:
            raise ValueError('filing_counts must be raw integer [B,6]')
        if filing_mask.device != filings.device or filing_counts.device != filings.device:
            raise ValueError('text inputs must be on the same device')
        if not torch.equal(filing_counts.long(), filing_mask.sum(-1)):
            raise ValueError('filing_counts must equal the number of valid filings')
        # Remove padding BEFORE projection; masked values cannot affect scores or gradients.
        clean = filings.masked_fill(~filing_mask.unsqueeze(-1), 0)
        projected = self.projection(clean)
        scores = self.filing_score(torch.tanh(self.filing_key(projected))).squeeze(-1)
        alpha = masked_softmax(scores, filing_mask)
        month_state = (alpha.unsqueeze(-1) * projected).sum(-2)
        month_mask = filing_mask.any(-1)
        aged_state = month_state + self.age_embedding(self.ages).unsqueeze(0)
        aged_state = aged_state.masked_fill(~month_mask.unsqueeze(-1), 0)
        time_scores = self.temporal_score(torch.tanh(self.temporal_key(aged_state))).squeeze(-1)
        beta = masked_softmax(time_scores, month_mask)
        h_text = (beta.unsqueeze(-1) * aged_state).sum(1)
        if return_diagnostics:
            return h_text, {'filing_attention': alpha, 'month_attention': beta,
                            'month_mask': month_mask, 'month_state': month_state,
                            'has_events': month_mask.any(-1), 'ages': self.ages}
        return h_text

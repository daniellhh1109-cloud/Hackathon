"""model model contracts, gradients and independent axis mixing checks."""
from dataclasses import replace
import pytest
import torch
from torch import nn
from src.models.modern_tcn import QuantModelConfig, ModernTCNBlock, ModernTCNEncoder
from src.models.quant_model import QuantRegressor, read_model_config, model_metadata
from src.training.trainer import seed_everything, load_checkpoint


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('batch', [1, 3, 16])
def test_shapes_backward_and_all_factor_access(batch):
    seed_everything(42)
    model = QuantRegressor()
    quant = torch.randn(batch,12,147,requires_grad=True)
    assert model.encoder(quant).shape == (batch,128)
    prediction = model(quant)
    assert prediction.shape == (batch,1)
    nn.HuberLoss()(prediction,torch.randn(batch,1)*.1).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert (quant.grad.abs().sum((0,1)) > 0).all()  # no factor silently sliced away
    assert (quant.grad.abs().sum((0,2)) > 0).all()  # mean pooling accesses all 12 months


def test_eval_independent_of_other_samples_and_order():
    seed_everything(5)
    model = QuantRegressor().eval()
    values = torch.randn(3,12,147)
    with torch.no_grad():
        first = model(values[:1])
        combined = model(values)
        reversed_batch = model(values.flip(0))
    torch.testing.assert_close(first, combined[:1], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(combined, reversed_batch.flip(0))
    assert not any(isinstance(module, nn.Embedding) for module in model.modules())


def test_residual_identity_when_branch_zero():
    block = ModernTCNBlock(dropout=0)
    with torch.no_grad():
        for p in block.parameters(): p.zero_()
    values = torch.randn(2,16,8,12)
    torch.testing.assert_close(block(values),values,rtol=0,atol=0)


def test_separate_group_mixing_axes():
    block = ModernTCNBlock(groups=3,features=2,dropout=0)
    with torch.no_grad():
        for module in (block.feature_ffn, block.variable_ffn):
            for layer in module:
                if isinstance(layer,nn.Conv1d):
                    layer.weight.fill_(1)
                    layer.bias.zero_()
    # Feature FFN mixes D only inside each M group.
    value = torch.zeros(1,6,1); value[0,0,0]=1
    result = block.feature_ffn(value).reshape(3,2)
    assert (result[0]>0).all() and (result[1:]==0).all()
    # Variable FFN receives D-major layout: mixes M within each D feature.
    result = block.variable_ffn(value).reshape(2,3)
    assert (result[0]>0).all() and (result[1]==0).all()
    assert block.temporal.groups == 6


@pytest.mark.parametrize('shape', [(12,147),(1,147,12),(1,11,147),(0,12,147)])
def test_bad_shapes(shape):
    with pytest.raises(ValueError,match='expects'):
        ModernTCNEncoder()(torch.zeros(shape))


def test_nan_and_integer_rejected():
    model = QuantRegressor()
    for value in (torch.full((1,12,147),float('nan')),torch.zeros(1,12,147,dtype=torch.long)):
        with pytest.raises(ValueError,match='finite floating'):
            model(value)


@pytest.mark.parametrize('change', [dict(kernel_size=6),dict(kernel_size=13),dict(latent_groups=7),
    dict(latent_groups=128),dict(dropout=1),dict(num_blocks=0),dict(pooling='bad')])
def test_invalid_config(change):
    with pytest.raises(ValueError): replace(QuantModelConfig(),**change)


def test_tiny_batch_learning_and_checkpoint(tmp_path):
    seed_everything(12)
    config = QuantModelConfig(dropout=0)
    model = QuantRegressor(config)
    quant = torch.randn(8,12,147)
    target = torch.tensor([-.1,.1,.02,-.04,.15,.06,-.15,.0]).reshape(-1,1)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
    initial = criterion(model(quant),target).item()
    for _ in range(80):
        optimizer.zero_grad()
        loss=criterion(model(quant),target)
        loss.backward()
        optimizer.step()
    model.eval()
    assert criterion(model(quant),target).item() < initial*.1
    names=[f'f{i}' for i in range(147)]
    path=tmp_path/'model.pt'
    torch.save({'format_version':1,'feature_names':names,'model_metadata':model_metadata(config),
                'model_state_dict':model.state_dict()},path)
    restored=QuantRegressor(config)
    load_checkpoint(path,restored,expected_feature_names=names,expected_model_metadata=model_metadata(config))
    torch.testing.assert_close(restored(quant),model(quant),rtol=0,atol=0)
    assert read_model_config('configs/model.yaml') == QuantModelConfig()

"""Shape and masking contracts for the three temporal architectures.

Every temporal model must accept (B, T, feature_dim) plus an optional
(B, T) mask and return (B, T, output_dim), so MultiTaskModel can wrap any of
them interchangeably.
"""

import pytest
import torch

from src.models.temporal import (
    BaselineModel, LSTMModel, MultiStageTCN, DilatedConvBlock, TCNStage,
)

FEAT_DIM = 64
HIDDEN = 32


def _models():
    return [
        ("baseline", BaselineModel(feature_dim=FEAT_DIM, hidden_dim=HIDDEN)),
        ("lstm", LSTMModel(feature_dim=FEAT_DIM, hidden_dim=HIDDEN,
                           num_layers=2, bidirectional=True)),
        ("ms_tcn", MultiStageTCN(feature_dim=FEAT_DIM, hidden_dim=HIDDEN,
                                 num_stages=2, num_layers=3, channels=16)),
    ]


@pytest.mark.parametrize("name,model", _models())
def test_output_shape_without_mask(name, model):
    out = model(torch.randn(2, 20, FEAT_DIM))
    assert out.shape == (2, 20, HIDDEN), f"{name} returned {tuple(out.shape)}"


@pytest.mark.parametrize("name,model", _models())
def test_output_shape_with_mask(name, model):
    """A mask must not change the output shape (padding is preserved)."""
    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[0, 15:] = False
    out = model(torch.randn(2, 20, FEAT_DIM), mask)
    assert out.shape == (2, 20, HIDDEN), f"{name} returned {tuple(out.shape)}"


@pytest.mark.parametrize("name,model", _models())
def test_output_dim_attribute_matches_output(name, model):
    """MultiTaskModel sizes its heads from output_dim, so it must be right."""
    out = model(torch.randn(1, 10, FEAT_DIM))
    assert model.output_dim == out.shape[-1]


@pytest.mark.parametrize("name,model", _models())
def test_gradients_flow(name, model):
    x = torch.randn(1, 12, FEAT_DIM, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_lstm_padding_does_not_leak_into_valid_steps():
    """Packing must make valid outputs independent of padded content."""
    model = LSTMModel(feature_dim=FEAT_DIM, hidden_dim=HIDDEN).eval()
    x = torch.randn(1, 10, FEAT_DIM)
    mask = torch.ones(1, 10, dtype=torch.bool)

    padded_x = torch.cat([x, torch.randn(1, 5, FEAT_DIM) * 100], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(1, 5, dtype=torch.bool)], dim=1)

    with torch.no_grad():
        base = model(x, mask)
        padded = model(padded_x, padded_mask)

    assert torch.allclose(base, padded[:, :10], atol=1e-5), (
        "LSTM output over valid steps changed when padding was appended"
    )


def test_tcn_zeroes_masked_positions():
    """MultiStageTCN masks padded steps after each stage."""
    model = MultiStageTCN(feature_dim=FEAT_DIM, hidden_dim=HIDDEN,
                          num_stages=2, num_layers=2, channels=16).eval()
    mask = torch.ones(1, 20, dtype=torch.bool)
    mask[0, 12:] = False
    with torch.no_grad():
        out = model(torch.randn(1, 20, FEAT_DIM), mask)
    # The projection has a bias, so masked positions share one constant value.
    tail = out[0, 12:]
    assert torch.allclose(tail, tail[0].expand_as(tail), atol=1e-6)


def test_dilated_block_preserves_shape_and_is_residual():
    block = DilatedConvBlock(channels=16, dilation=4, dropout=0.0).eval()
    x = torch.randn(1, 16, 30)
    with torch.no_grad():
        out = block(x)
    assert out.shape == x.shape
    assert not torch.allclose(out, x), "residual block collapsed to identity"


def test_tcn_stage_projects_channels():
    stage = TCNStage(input_dim=FEAT_DIM, channels=16, num_layers=3)
    assert stage(torch.randn(1, FEAT_DIM, 25)).shape == (1, 16, 25)


def test_tcn_receptive_field_grows_with_layers():
    """Dilations double per layer, so a later layer sees strictly further."""
    stage = TCNStage(input_dim=FEAT_DIM, channels=8, num_layers=4)
    dilations = [b.conv.dilation[0] for b in stage.layers]
    assert dilations == [1, 2, 4, 8]


def test_baseline_has_no_temporal_context():
    """The baseline must treat each frame independently."""
    model = BaselineModel(feature_dim=FEAT_DIM, hidden_dim=HIDDEN).eval()
    x = torch.randn(1, 10, FEAT_DIM)
    with torch.no_grad():
        full = model(x)
        # Changing a later frame must not affect an earlier output.
        perturbed = x.clone()
        perturbed[0, 5:] = torch.randn(5, FEAT_DIM)
        assert torch.allclose(full[0, :5], model(perturbed)[0, :5], atol=1e-6)

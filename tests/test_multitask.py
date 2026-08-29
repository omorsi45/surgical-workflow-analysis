import torch
import pytest
from src.models.multitask import CorrelationLoss


def _make_loss():
    cooccur = torch.rand(7, 7)
    return CorrelationLoss(cooccur)


def test_corr_loss_phase_gradients_nonzero():
    """Phase logits must receive gradient through CorrelationLoss."""
    loss_fn = _make_loss()
    phase_logits = torch.randn(1, 10, 7, requires_grad=True)
    tool_logits = torch.randn(1, 10, 7, requires_grad=True)
    loss = loss_fn(phase_logits, tool_logits)
    loss.backward()
    assert phase_logits.grad is not None
    assert phase_logits.grad.abs().sum().item() > 0, (
        "phase_logits has zero gradient — argmax is non-differentiable"
    )


def test_corr_loss_tool_gradients_nonzero():
    """Tool logits must also receive gradient."""
    loss_fn = _make_loss()
    phase_logits = torch.randn(1, 10, 7, requires_grad=True)
    tool_logits = torch.randn(1, 10, 7, requires_grad=True)
    loss = loss_fn(phase_logits, tool_logits)
    loss.backward()
    assert tool_logits.grad is not None
    assert tool_logits.grad.abs().sum().item() > 0


def test_corr_loss_mask_respected():
    """Loss with mask should differ from loss without."""
    torch.manual_seed(0)
    loss_fn = _make_loss()
    phase_logits = torch.randn(2, 20, 7)
    tool_logits = torch.randn(2, 20, 7)
    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[0, 10:] = False

    loss_with_mask = loss_fn(phase_logits, tool_logits, mask)
    loss_no_mask = loss_fn(phase_logits, tool_logits, None)
    assert not torch.isclose(loss_with_mask, loss_no_mask)


import numpy as np
from src.models.multitask import (
    MultiTaskLoss, MultiTaskModel, build_cooccurrence_matrix,
)
from src.models.temporal import BaselineModel


# ---------------------------------------------------------------------------
# Loss normalization
#
# The masked path summed over (B, T, num_tools) but divided by mask.sum(),
# which counts frames only — inflating the tool and correlation terms by
# num_tools and silently rescaling lambda_tool / lambda_corr.
# ---------------------------------------------------------------------------


def _loss_inputs(B=2, T=15, C=7, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(B, T, C),
        torch.randn(B, T, C),
        torch.randint(0, 7, (B, T)),
        torch.randint(0, 2, (B, T, C)).float(),
    )


def test_tool_loss_masked_matches_unmasked_when_all_valid():
    """An all-True mask must give the same tool loss as no mask at all."""
    pl, tl, pt, tt = _loss_inputs()
    loss_fn = MultiTaskLoss(torch.ones(7))
    full_mask = torch.ones(pl.shape[:2], dtype=torch.bool)

    _, masked = loss_fn(pl, tl, pt, tt, full_mask)
    _, unmasked = loss_fn(pl, tl, pt, tt, None)

    assert abs(masked["tool"] - unmasked["tool"]) < 1e-6, (
        f"tool loss {masked['tool']:.6f} (masked) vs {unmasked['tool']:.6f} "
        "(unmasked) — masked path is not normalized per element"
    )


def test_corr_loss_masked_matches_unmasked_when_all_valid():
    """Same invariant for the correlation term."""
    pl, tl, pt, tt = _loss_inputs()
    loss_fn = MultiTaskLoss(torch.ones(7), torch.rand(7, 7), lambda_corr=0.5)
    full_mask = torch.ones(pl.shape[:2], dtype=torch.bool)

    _, masked = loss_fn(pl, tl, pt, tt, full_mask)
    _, unmasked = loss_fn(pl, tl, pt, tt, None)

    assert abs(masked["corr"] - unmasked["corr"]) < 1e-6


def test_tool_loss_ignores_padded_frames():
    """Padding must not change the tool loss on the valid region."""
    pl, tl, pt, tt = _loss_inputs(B=1, T=10)
    loss_fn = MultiTaskLoss(torch.ones(7))

    full_mask = torch.ones(1, 10, dtype=torch.bool)
    _, unpadded = loss_fn(pl, tl, pt, tt, full_mask)

    # Append 5 padded steps carrying wildly different logits.
    pad_logits = torch.full((1, 5, 7), 50.0)
    pl_pad = torch.cat([pl, pad_logits], dim=1)
    tl_pad = torch.cat([tl, pad_logits], dim=1)
    pt_pad = torch.cat([pt, torch.full((1, 5), -1, dtype=torch.long)], dim=1)
    tt_pad = torch.cat([tt, torch.zeros(1, 5, 7)], dim=1)
    mask = torch.cat([full_mask, torch.zeros(1, 5, dtype=torch.bool)], dim=1)

    _, padded = loss_fn(pl_pad, tl_pad, pt_pad, tt_pad, mask)

    assert abs(padded["tool"] - unpadded["tool"]) < 1e-6
    assert abs(padded["phase"] - unpadded["phase"]) < 1e-5, (
        "phase loss changed with padding — ignore_index=-1 not taking effect"
    )


def test_lambda_weights_scale_total_loss():
    """Total must be the documented weighted sum of the three components."""
    pl, tl, pt, tt = _loss_inputs()
    mask = torch.ones(pl.shape[:2], dtype=torch.bool)
    loss_fn = MultiTaskLoss(torch.ones(7), torch.rand(7, 7),
                            lambda_phase=1.0, lambda_tool=2.0, lambda_corr=0.5)
    _, d = loss_fn(pl, tl, pt, tt, mask)
    expected = 1.0 * d["phase"] + 2.0 * d["tool"] + 0.5 * d["corr"]
    assert abs(d["total"] - expected) < 1e-5


# ---------------------------------------------------------------------------
# Co-occurrence prior
# ---------------------------------------------------------------------------


def _write_feature_file(tmp_path, vid, phases, tools, feat_dim=4):
    torch.save(
        {
            "features": torch.randn(len(phases), feat_dim),
            "phases": torch.tensor(phases, dtype=torch.long),
            "tools": torch.tensor(tools, dtype=torch.float32),
        },
        tmp_path / f"video{vid:02d}.pt",
    )


def test_cooccurrence_matches_hand_computed_probability(tmp_path):
    """P(tool | phase) must be the empirical presence rate."""
    # Phase 0 x 4 frames; Grasper present in 3 of them.
    phases = [0, 0, 0, 0]
    tools = [[1, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0]]
    _write_feature_file(tmp_path, 1, phases, tools)

    m = build_cooccurrence_matrix(str(tmp_path), [1])
    assert m.shape == (7, 7)
    assert abs(m[0, 0].item() - 0.75) < 1e-6
    assert m[0, 1].item() == 0.0


def test_laplace_smoothing_removes_zero_entries(tmp_path):
    """Smoothing must keep every prior strictly inside (0, 1)."""
    phases = [0, 0, 0, 0]
    tools = [[1, 0, 0, 0, 0, 0, 0]] * 4
    _write_feature_file(tmp_path, 1, phases, tools)

    unsmoothed = build_cooccurrence_matrix(str(tmp_path), [1])
    smoothed = build_cooccurrence_matrix(str(tmp_path), [1], smoothing=1.0)

    assert (unsmoothed == 0.0).any(), "expected zeros without smoothing"
    assert (smoothed > 0.0).all(), "smoothing must eliminate exact zeros"
    assert (smoothed < 1.0).all(), "smoothing must eliminate exact ones"


def test_cooccurrence_probabilities_in_range(tmp_path):
    phases = [0, 1, 2, 3, 4, 5, 6]
    tools = np.random.randint(0, 2, (7, 7)).tolist()
    _write_feature_file(tmp_path, 1, phases, tools)
    m = build_cooccurrence_matrix(str(tmp_path), [1], smoothing=0.5)
    assert ((m >= 0.0) & (m <= 1.0)).all()


def test_multitask_model_output_shapes():
    model = MultiTaskModel(BaselineModel(feature_dim=32, hidden_dim=16))
    phase_logits, tool_logits = model(torch.randn(2, 9, 32))
    assert phase_logits.shape == (2, 9, 7)
    assert tool_logits.shape == (2, 9, 7)

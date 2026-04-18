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

import torch
import numpy as np
from src.evaluate import compute_tool_map, compute_per_tool_ap


def _make_predictions(n=100, num_tools=7, seed=0):
    torch.manual_seed(seed)
    preds = torch.rand(n, num_tools)
    targets = torch.randint(0, 2, (n, num_tools)).float()
    targets[0, :] = 1.0  # ensure at least one positive per tool
    return preds, targets


def test_compute_tool_map_equals_mean_per_tool_ap():
    """compute_tool_map must equal compute_per_tool_ap().mean()."""
    preds, targets = _make_predictions()
    map_direct = compute_tool_map(preds, targets)
    map_via_per = float(compute_per_tool_ap(preds, targets).mean())
    assert abs(map_direct - map_via_per) < 1e-6, (
        f"compute_tool_map={map_direct:.6f} != "
        f"compute_per_tool_ap().mean()={map_via_per:.6f}"
    )


def test_per_tool_ap_shape():
    preds, targets = _make_predictions()
    aps = compute_per_tool_ap(preds, targets)
    assert aps.shape == (7,)
    assert all(0.0 <= ap <= 1.0 for ap in aps)

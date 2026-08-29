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


from src.evaluate import (
    compute_phase_f1, compute_per_phase_f1, compute_phase_accuracy,
    compute_edit_score,
)


# ---------------------------------------------------------------------------
# mAP with classes absent from the split
#
# AP is undefined for a class with no positives. Scoring it 0.0 penalized the
# model for a tool it was never evaluated on and dragged mAP down.
# ---------------------------------------------------------------------------


def test_absent_tool_gets_nan_not_zero():
    preds, targets = _make_predictions()
    targets[:, 3] = 0.0  # Scissors never present
    aps = compute_per_tool_ap(preds, targets)
    assert np.isnan(aps[3]), "absent tool must be nan, not 0.0"
    assert not np.isnan(np.delete(aps, 3)).any()


def test_map_excludes_absent_tools():
    preds, targets = _make_predictions()
    targets[:, 3] = 0.0
    aps = compute_per_tool_ap(preds, targets)

    mAP = compute_tool_map(preds, targets)
    assert abs(mAP - np.nanmean(aps)) < 1e-9
    # Scoring the absent tool as 0.0 would give a strictly lower value.
    assert mAP > float(np.nan_to_num(aps).mean())


def test_map_all_tools_absent_returns_zero():
    preds, targets = _make_predictions()
    targets[:] = 0.0
    assert compute_tool_map(preds, targets) == 0.0


# ---------------------------------------------------------------------------
# Phase metrics
# ---------------------------------------------------------------------------


def test_phase_f1_perfect_prediction():
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    assert abs(compute_phase_f1(labels, labels) - 1.0) < 1e-9


def test_phase_f1_macro_counts_all_classes():
    """Macro F1 must average over all 7 phases, including unpredicted ones."""
    preds = torch.tensor([0, 0, 0, 0])
    targets = torch.tensor([0, 0, 0, 0])
    # Only phase 0 occurs, so 6 of 7 classes score 0 -> macro F1 = 1/7.
    assert abs(compute_phase_f1(preds, targets) - 1.0 / 7.0) < 1e-9


def test_per_phase_f1_shape_and_range():
    preds = torch.randint(0, 7, (200,))
    targets = torch.randint(0, 7, (200,))
    f1s = compute_per_phase_f1(preds, targets)
    assert f1s.shape == (7,)
    assert ((f1s >= 0.0) & (f1s <= 1.0)).all()


def test_phase_accuracy():
    preds = torch.tensor([0, 1, 2, 1, 0])
    targets = torch.tensor([0, 1, 1, 1, 0])
    assert abs(compute_phase_accuracy(preds, targets) - 0.8) < 1e-9


def test_phase_accuracy_empty_sequence():
    empty = torch.tensor([], dtype=torch.long)
    assert compute_phase_accuracy(empty, empty) == 0.0


# ---------------------------------------------------------------------------
# Edit score (temporal consistency)
# ---------------------------------------------------------------------------


def test_edit_score_identical_sequences():
    seq = torch.tensor([0, 0, 1, 1, 2, 2])
    assert abs(compute_edit_score(seq, seq) - 1.0) < 1e-9


def test_edit_score_ignores_segment_duration():
    """Run-length encoding means only the segment *order* matters."""
    a = torch.tensor([0, 0, 0, 0, 1, 1])
    b = torch.tensor([0, 1, 1, 1, 1, 1])
    assert abs(compute_edit_score(a, b) - 1.0) < 1e-9


def test_edit_score_penalizes_flicker():
    """A flickering prediction must score below a clean one."""
    truth = torch.tensor([0] * 6 + [1] * 6)
    clean = torch.tensor([0] * 6 + [1] * 6)
    flicker = torch.tensor([0, 1] * 6)
    assert compute_edit_score(flicker, truth) < compute_edit_score(clean, truth)


def test_edit_score_bounds():
    truth = torch.tensor([0] * 10 + [1] * 10)
    flicker = torch.tensor([0, 1] * 10)
    for score in (compute_edit_score(truth, truth),
                  compute_edit_score(flicker, truth)):
        assert 0.0 <= score <= 1.0


def test_edit_score_empty_sequences():
    empty = torch.tensor([], dtype=torch.long)
    assert compute_edit_score(empty, empty) == 1.0

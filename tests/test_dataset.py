import numpy as np
from unittest.mock import patch, MagicMock
from src.dataset import Cholec80VideoDataset


def _make_dataset_with_mp4(tmp_path):
    """Minimal dataset backed by a fake MP4 with 3 annotated frames."""
    frame_nums = [0, 25, 50]
    phase_file = tmp_path / "video01-phase.txt"
    phase_file.write_text("Frame\tPhase\n" + "\n".join(f"{fn}\t0" for fn in frame_nums))
    tool_file = tmp_path / "video01-tool.txt"
    tool_file.write_text(
        "Frame\tGrasper\tBipolar\tHook\tScissors\tClipper\tIrrigator\tSpecimenBag\n"
        + "\n".join(f"{fn}\t1\t0\t0\t0\t0\t0\t0" for fn in frame_nums)
    )
    mp4_file = tmp_path / "videos" / "video01.mp4"
    mp4_file.parent.mkdir()
    mp4_file.touch()

    fake_frame = np.zeros((224, 224, 3), dtype=np.uint8)
    cap = MagicMock()
    cap.read.return_value = (True, fake_frame)

    with patch("src.dataset.glob.glob", return_value=[]):
        with patch("cv2.VideoCapture", return_value=cap):
            ds = Cholec80VideoDataset(str(tmp_path), video_id=1)
    return ds, frame_nums


def test_mp4_frame_cache_populated(tmp_path):
    """All frames must be pre-loaded into _frame_cache at init."""
    ds, frame_nums = _make_dataset_with_mp4(tmp_path)
    assert hasattr(ds, "_frame_cache"), "_frame_cache must exist after init"
    for fn in frame_nums:
        assert fn in ds._frame_cache, f"Frame {fn} missing from cache"


def test_mp4_videocapture_not_opened_in_getitem(tmp_path):
    """VideoCapture must not be opened during __getitem__ calls."""
    ds, _ = _make_dataset_with_mp4(tmp_path)
    with patch("cv2.VideoCapture") as mock_cap:
        _ = ds[0]
        _ = ds[1]
        mock_cap.assert_not_called()


import pytest
import torch

from src.dataset import (
    collate_sequences, parse_phase_annotations, parse_tool_annotations,
    PHASE_NAME_TO_IDX,
)


# ---------------------------------------------------------------------------
# collate_sequences — padding and mask semantics
#
# The whole masking contract downstream (loss, metrics, LSTM packing) rests on
# phases being padded with -1 and mask marking exactly the valid steps.
# ---------------------------------------------------------------------------


def _item(length, feat_dim=8):
    return (
        torch.randn(length, feat_dim),
        torch.randint(0, 7, (length,)),
        torch.randint(0, 2, (length, 7)).float(),
    )


def test_collate_pads_to_longest_sequence():
    feats, phases, tools, mask = collate_sequences([_item(10), _item(25)])
    assert feats.shape == (2, 25, 8)
    assert phases.shape == (2, 25)
    assert tools.shape == (2, 25, 7)
    assert mask.shape == (2, 25)


def test_collate_mask_marks_exactly_the_valid_steps():
    feats, phases, tools, mask = collate_sequences([_item(10), _item(25)])
    assert mask[0].sum().item() == 10
    assert mask[1].sum().item() == 25
    assert mask[0, :10].all() and not mask[0, 10:].any()


def test_collate_pads_phases_with_ignore_index():
    """Phase padding must be -1 so CrossEntropyLoss(ignore_index=-1) skips it."""
    _, phases, _, mask = collate_sequences([_item(10), _item(25)])
    assert (phases[0, 10:] == -1).all(), "padded phases must be -1"
    assert (phases[~mask] == -1).all()
    assert (phases[mask] >= 0).all(), "valid phases must never be -1"


def test_collate_pads_tools_and_features_with_zero():
    feats, _, tools, mask = collate_sequences([_item(10), _item(25)])
    assert (tools[0, 10:] == 0).all()
    assert (feats[0, 10:] == 0).all()


def test_collate_preserves_content():
    a, b = _item(6), _item(9)
    feats, phases, tools, _ = collate_sequences([a, b])
    assert torch.allclose(feats[0, :6], a[0])
    assert torch.equal(phases[1, :9], b[1])
    assert torch.allclose(tools[0, :6], a[2])


def test_collate_equal_lengths_needs_no_padding():
    _, _, _, mask = collate_sequences([_item(12), _item(12)])
    assert mask.all()


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------


def test_parse_phase_annotations_named_labels(tmp_path):
    f = tmp_path / "video01-phase.txt"
    f.write_text("Frame\tPhase\n0\tPreparation\n25\tCalotTriangleDissection\n")
    phases = parse_phase_annotations(str(f))
    assert phases == {0: 0, 25: 1}


def test_parse_phase_annotations_integer_labels(tmp_path):
    f = tmp_path / "video01-phase.txt"
    f.write_text("Frame\tPhase\n0\t0\n25\t3\n")
    assert parse_phase_annotations(str(f)) == {0: 0, 25: 3}


def test_parse_phase_annotations_rejects_unknown_label(tmp_path):
    """An unrecognized name must raise, not silently become Preparation."""
    f = tmp_path / "video01-phase.txt"
    f.write_text("Frame\tPhase\n0\tNotARealPhase\n")
    with pytest.raises(ValueError, match="Unrecognized phase label"):
        parse_phase_annotations(str(f))


def test_phase_name_map_covers_both_spellings():
    """Compact and spaced spellings must agree on the index."""
    for compact, spaced in [
        ("CalotTriangleDissection", "Calot Triangle Dissection"),
        ("ClippingCutting", "Clipping and Cutting"),
        ("GallbladderDissection", "Gallbladder Dissection"),
        ("GallbladderPackaging", "Gallbladder Packaging"),
        ("CleaningCoagulation", "Cleaning and Coagulation"),
        ("GallbladderRetraction", "Gallbladder Retraction"),
    ]:
        assert PHASE_NAME_TO_IDX[compact] == PHASE_NAME_TO_IDX[spaced]


def test_parse_tool_annotations(tmp_path):
    f = tmp_path / "video01-tool.txt"
    f.write_text(
        "Frame\tGrasper\tBipolar\tHook\tScissors\tClipper\tIrrigator\tSpecimenBag\n"
        "0\t1\t0\t0\t0\t0\t0\t0\n25\t0\t1\t1\t0\t0\t0\t0\n"
    )
    tools = parse_tool_annotations(str(f))
    assert list(tools[0]) == [1, 0, 0, 0, 0, 0, 0]
    assert list(tools[25]) == [0, 1, 1, 0, 0, 0, 0]


def test_mp4_path_rejects_unsupported_fps(tmp_path):
    """Tool annotations only exist at 1 fps; other rates must raise."""
    frame_nums = [0, 25, 50]
    (tmp_path / "video01-phase.txt").write_text(
        "Frame\tPhase\n" + "\n".join(f"{fn}\t0" for fn in frame_nums))
    (tmp_path / "video01-tool.txt").write_text(
        "Frame\tGrasper\tBipolar\tHook\tScissors\tClipper\tIrrigator\tSpecimenBag\n"
        + "\n".join(f"{fn}\t1\t0\t0\t0\t0\t0\t0" for fn in frame_nums))
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "video01.mp4").touch()

    with patch("src.dataset.glob.glob", return_value=[]):
        with pytest.raises(ValueError, match="not supported"):
            Cholec80VideoDataset(str(tmp_path), video_id=1, fps=5)

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

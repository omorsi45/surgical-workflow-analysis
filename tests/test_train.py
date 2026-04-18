import torch
from src.utils import AverageMeter


def test_average_meter_sample_weighted():
    """AverageMeter must weight by n, not just count updates."""
    meter = AverageMeter()
    meter.update(1.0, n=4)
    meter.update(3.0, n=4)
    assert abs(meter.avg - 2.0) < 1e-6, f"Expected 2.0, got {meter.avg}"

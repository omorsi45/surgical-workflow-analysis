import torch
import torch.nn as nn
from unittest.mock import MagicMock
from src.utils import AverageMeter
from src.train import Trainer


def test_average_meter_sample_weighted():
    """AverageMeter must weight by n, not just count updates."""
    meter = AverageMeter()
    meter.update(1.0, n=4)
    meter.update(3.0, n=4)
    assert abs(meter.avg - 2.0) < 1e-6, f"Expected 2.0, got {meter.avg}"


def _make_minimal_trainer():
    model = nn.Linear(4, 2)
    loss_fn = MagicMock()
    loss_fn.to = MagicMock(return_value=loss_fn)
    config = {
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "grad_clip_max_norm": 5.0,
            "scheduler_T0": 10,
            "early_stopping_patience": 5,
            "num_epochs": 1,
        },
        "data": {"num_phases": 7},
    }
    return Trainer(model, loss_fn, [], [], config, save_dir="/tmp/test_ckpt", device="cpu")


def test_trainer_uses_adamw():
    trainer = _make_minimal_trainer()
    assert isinstance(trainer.optimizer, torch.optim.AdamW), (
        f"Expected AdamW, got {type(trainer.optimizer)}"
    )


def test_trainer_has_grad_scaler():
    trainer = _make_minimal_trainer()
    assert hasattr(trainer, "scaler"), "Trainer must have a GradScaler for AMP"
    assert isinstance(trainer.scaler, torch.amp.GradScaler)

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


# ---------------------------------------------------------------------------
# Best-checkpoint restore
#
# Early stopping selects a model on validation F1 and saves it, but the model
# left in memory after train() used to be the *last* epoch's — `patience`
# epochs past the selected one. Test metrics must come from the selected model.
# ---------------------------------------------------------------------------

from torch.utils.data import DataLoader

from src.dataset import collate_sequences
from src.models.multitask import MultiTaskModel, MultiTaskLoss
from src.models.temporal import BaselineModel


def _tiny_loader(num_videos=2, length=12, feat_dim=8, seed=0):
    """Two short synthetic 'videos' shaped like Cholec80FeatureDataset items."""
    torch.manual_seed(seed)
    items = [
        (
            torch.randn(length, feat_dim),
            torch.randint(0, 7, (length,)),
            torch.randint(0, 2, (length, 7)).float(),
        )
        for _ in range(num_videos)
    ]
    return DataLoader(items, batch_size=1, collate_fn=collate_sequences)


def _tiny_trainer(tmp_path, num_epochs=6, patience=2, feat_dim=8):
    config = {
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "grad_clip_max_norm": 5.0,
            "scheduler_T0": 10,
            "early_stopping_patience": patience,
            "num_epochs": num_epochs,
        },
        "data": {"num_phases": 7, "num_tools": 7},
    }
    model = MultiTaskModel(BaselineModel(feature_dim=feat_dim, hidden_dim=16))
    loss_fn = MultiTaskLoss(torch.ones(7))
    return Trainer(
        model, loss_fn, _tiny_loader(feat_dim=feat_dim),
        _tiny_loader(feat_dim=feat_dim, seed=1),
        config, save_dir=str(tmp_path / "ckpt"), device="cpu",
    )


def test_train_restores_best_checkpoint(tmp_path):
    """After train(), in-memory weights must equal the saved best checkpoint."""
    trainer = _tiny_trainer(tmp_path)
    trainer.train()

    saved = torch.load(tmp_path / "ckpt" / "best_model.pt",
                       map_location="cpu", weights_only=False)["model_state_dict"]
    live = trainer.model.state_dict()

    for key, saved_tensor in saved.items():
        assert torch.allclose(live[key].cpu(), saved_tensor.cpu()), (
            f"Parameter {key!r} differs from the best checkpoint — train() "
            "left last-epoch weights in memory instead of restoring the "
            "model that early stopping selected."
        )


def test_load_best_checkpoint_returns_false_without_file(tmp_path):
    """Missing checkpoint must be reported, not raise."""
    trainer = _tiny_trainer(tmp_path)
    assert trainer.load_best_checkpoint() is False


def test_best_val_f1_matches_history(tmp_path):
    """The restored checkpoint should carry the best F1 seen during training."""
    trainer = _tiny_trainer(tmp_path)
    history = trainer.train()
    assert abs(trainer.best_val_f1 - max(history["val_phase_f1"])) < 1e-9

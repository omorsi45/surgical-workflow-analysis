"""
Multi-task model wrapper and correlation loss for surgical workflow analysis.

Combines a temporal model with two task-specific heads (phase classification
and tool detection). Includes a novel correlation loss that penalizes
tool predictions inconsistent with the predicted surgical phase.

Author: Omar Morsi (40236376)
"""

import os
import torch
import torch.nn as nn
import numpy as np


class MultiTaskModel(nn.Module):
    """Multi-task wrapper that adds phase and tool heads to any temporal model.

    Takes a temporal model (BaselineModel, LSTMModel, or MultiStageTCN)
    and attaches two classification heads: one for phase recognition
    (7-class softmax) and one for tool detection (7-class sigmoid).

    Args:
        temporal_model (nn.Module): A temporal model with output_dim attribute.
        num_phases (int): Number of surgical phases. Default: 7.
        num_tools (int): Number of surgical tools. Default: 7.

    Example:
        >>> from src.models.temporal import MultiStageTCN
        >>> tcn = MultiStageTCN(feature_dim=2048, hidden_dim=512)
        >>> model = MultiTaskModel(tcn, num_phases=7, num_tools=7)
        >>> x = torch.randn(1, 100, 2048)
        >>> phase_logits, tool_logits = model(x)
        >>> print(phase_logits.shape, tool_logits.shape)
        torch.Size([1, 100, 7]) torch.Size([1, 100, 7])
    """

    def __init__(self, temporal_model, num_phases=7, num_tools=7):
        super().__init__()
        self.temporal_model = temporal_model
        hidden_dim = temporal_model.output_dim

        self.phase_head = nn.Linear(hidden_dim, num_phases)
        self.tool_head = nn.Linear(hidden_dim, num_tools)

    def forward(self, x, mask=None):
        """Forward pass through temporal model and both task heads.

        Args:
            x (torch.Tensor): Input features, shape (B, T, feature_dim).
            mask (torch.BoolTensor, optional): Valid frame mask, shape (B, T).

        Returns:
            tuple: (phase_logits, tool_logits) where:
                - phase_logits: shape (B, T, num_phases)
                - tool_logits: shape (B, T, num_tools)
        """
        hidden = self.temporal_model(x, mask)
        phase_logits = self.phase_head(hidden)
        tool_logits = self.tool_head(hidden)
        return phase_logits, tool_logits


def build_cooccurrence_matrix(features_dir, video_ids, num_phases=7, num_tools=7):
    """Build a tool-phase co-occurrence probability matrix from training data.

    Counts how often each tool appears during each phase across the training
    set, then normalizes each row to get P(tool | phase).

    Args:
        features_dir (str): Directory with cached .pt feature files.
        video_ids (list): List of training video IDs.
        num_phases (int): Number of surgical phases. Default: 7.
        num_tools (int): Number of surgical tools. Default: 7.

    Returns:
        torch.Tensor: Co-occurrence probability matrix of shape
            (num_phases, num_tools), where entry [i, j] = P(tool_j | phase_i).

    Example:
        >>> cooccur = build_cooccurrence_matrix("data/features", [1, 2, 3])
        >>> print(cooccur.shape)
        torch.Size([7, 7])
        >>> print(cooccur[0])  # Tool probabilities during Preparation phase
    """
    counts = torch.zeros(num_phases, num_tools)
    phase_counts = torch.zeros(num_phases)

    for vid in video_ids:
        data = torch.load(
            os.path.join(features_dir, f"video{vid:02d}.pt"),
            weights_only=True,
        )
        phases = data["phases"]
        tools = data["tools"]
        for p in range(num_phases):
            mask = phases == p
            phase_counts[p] += mask.sum()
            counts[p] += tools[mask].sum(dim=0)

    # Normalize: P(tool | phase)
    phase_counts = phase_counts.clamp(min=1)
    cooccur = counts / phase_counts.unsqueeze(1)
    return cooccur


class CorrelationLoss(nn.Module):
    """Novel correlation loss that penalizes impossible tool-phase combinations.

    Uses a pre-computed co-occurrence matrix P(tool | phase) to penalize
    the model when it predicts high probability for tools that are rare
    during the predicted surgical phase.

    The loss is: L_corr = mean(tool_probs * (1 - prior_probs))
    This is high when the model predicts a tool (tool_probs near 1.0) but
    the prior says that tool never appears in the current phase
    (prior_probs near 0.0).

    Args:
        cooccurrence_matrix (torch.Tensor): Shape (num_phases, num_tools),
            where entry [i, j] = P(tool_j | phase_i).

    Example:
        >>> cooccur = torch.rand(7, 7)
        >>> loss_fn = CorrelationLoss(cooccur)
        >>> phase_logits = torch.randn(1, 100, 7)
        >>> tool_logits = torch.randn(1, 100, 7)
        >>> mask = torch.ones(1, 100, dtype=torch.bool)
        >>> loss = loss_fn(phase_logits, tool_logits, mask)
        >>> print(loss.shape)
        torch.Size([])
    """

    def __init__(self, cooccurrence_matrix):
        super().__init__()
        # Register as buffer so it moves to GPU with the model
        self.register_buffer("cooccur", cooccurrence_matrix)

    def forward(self, phase_logits, tool_logits, mask=None):
        """Compute the correlation loss.

        Args:
            phase_logits (torch.Tensor): Phase predictions, shape (B, T, num_phases).
            tool_logits (torch.Tensor): Tool predictions, shape (B, T, num_tools).
            mask (torch.BoolTensor, optional): Valid frame mask, shape (B, T).

        Returns:
            torch.Tensor: Scalar correlation loss.
        """
        # Soft phase assignment — differentiable w.r.t. phase_logits
        phase_probs = torch.softmax(phase_logits, dim=-1)       # (B, T, num_phases)
        tool_probs = torch.sigmoid(tool_logits)                  # (B, T, num_tools)

        # Expected prior: weighted average of co-occurrence rows
        prior_probs = phase_probs @ self.cooccur                 # (B, T, num_tools)

        # Penalty: high tool prob where prior is low
        penalty = tool_probs * (1.0 - prior_probs)

        if mask is not None:
            penalty = penalty * mask.unsqueeze(-1).float()
            return penalty.sum() / mask.sum().clamp(min=1)

        return penalty.mean()


class MultiTaskLoss(nn.Module):
    """Combined multi-task loss with phase, tool, and correlation components.

    Computes the weighted sum:
        L = lambda_phase * WeightedCE(phase) + lambda_tool * BCE(tool)
            + lambda_corr * CorrelationLoss(phase, tool)

    Args:
        phase_weights (torch.Tensor): Class weights for phase CE loss,
            shape (num_phases,).
        cooccurrence_matrix (torch.Tensor, optional): For correlation loss,
            shape (num_phases, num_tools). If None, correlation loss is disabled.
        lambda_phase (float): Weight for phase loss. Default: 1.0.
        lambda_tool (float): Weight for tool loss. Default: 1.0.
        lambda_corr (float): Weight for correlation loss. Default: 0.5.

    Example:
        >>> weights = torch.ones(7)
        >>> cooccur = torch.rand(7, 7)
        >>> loss_fn = MultiTaskLoss(weights, cooccur)
        >>> phase_logits = torch.randn(1, 100, 7)
        >>> tool_logits = torch.randn(1, 100, 7)
        >>> phase_targets = torch.randint(0, 7, (1, 100))
        >>> tool_targets = torch.randint(0, 2, (1, 100, 7)).float()
        >>> mask = torch.ones(1, 100, dtype=torch.bool)
        >>> total, details = loss_fn(phase_logits, tool_logits,
        ...                          phase_targets, tool_targets, mask)
    """

    def __init__(self, phase_weights, cooccurrence_matrix=None,
                 lambda_phase=1.0, lambda_tool=1.0, lambda_corr=0.5):
        super().__init__()
        self.lambda_phase = lambda_phase
        self.lambda_tool = lambda_tool
        self.lambda_corr = lambda_corr

        self.phase_criterion = nn.CrossEntropyLoss(
            weight=phase_weights, ignore_index=-1
        )
        self.tool_criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.corr_loss = None
        if cooccurrence_matrix is not None:
            self.corr_loss = CorrelationLoss(cooccurrence_matrix)

    def forward(self, phase_logits, tool_logits, phase_targets, tool_targets,
                mask=None):
        """Compute total multi-task loss.

        Args:
            phase_logits (torch.Tensor): Shape (B, T, num_phases).
            tool_logits (torch.Tensor): Shape (B, T, num_tools).
            phase_targets (torch.Tensor): Shape (B, T), int labels 0-6.
            tool_targets (torch.Tensor): Shape (B, T, num_tools), binary.
            mask (torch.BoolTensor, optional): Shape (B, T).

        Returns:
            tuple: (total_loss, loss_dict) where loss_dict has keys
                "phase", "tool", "corr", "total".
        """
        B, T, C_phase = phase_logits.shape

        # Phase loss: reshape for CrossEntropyLoss (expects (N, C))
        phase_loss = self.phase_criterion(
            phase_logits.reshape(-1, C_phase), phase_targets.reshape(-1)
        )

        # Tool loss: apply mask manually
        tool_loss_raw = self.tool_criterion(tool_logits, tool_targets)
        if mask is not None:
            tool_loss_raw = tool_loss_raw * mask.unsqueeze(-1).float()
            tool_loss = tool_loss_raw.sum() / mask.sum().clamp(min=1)
        else:
            tool_loss = tool_loss_raw.mean()

        total = self.lambda_phase * phase_loss + self.lambda_tool * tool_loss

        corr_loss = torch.tensor(0.0, device=phase_logits.device)
        if self.corr_loss is not None and self.lambda_corr > 0:
            corr_loss = self.corr_loss(phase_logits, tool_logits, mask)
            total = total + self.lambda_corr * corr_loss

        loss_dict = {
            "phase": phase_loss.item(),
            "tool": tool_loss.item(),
            "corr": corr_loss.item(),
            "total": total.item(),
        }
        return total, loss_dict

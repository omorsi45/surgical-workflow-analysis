"""
Temporal models for surgical workflow analysis.

Implements three temporal architectures that consume sequences of pre-extracted
ResNet-50 features and produce hidden representations for multi-task heads:
1. BaselineModel -- frame-wise FC, no temporal modeling
2. LSTMModel -- bidirectional LSTM for sequence modeling
3. MultiStageTCN -- multi-stage temporal convolutional network with dilated
   convolutions for long-range temporal dependencies

Author: Omar Morsi (40236376)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineModel(nn.Module):
    """Frame-wise baseline with no temporal modeling.

    Processes each frame independently through a fully-connected layer.
    Serves as a lower-bound baseline to quantify the benefit of temporal models.

    Args:
        feature_dim (int): Input feature dimension (from ResNet-50). Default: 2048.
        hidden_dim (int): Output hidden dimension. Default: 512.
        dropout (float): Dropout probability. Default: 0.3.

    Example:
        >>> model = BaselineModel(feature_dim=2048, hidden_dim=512)
        >>> x = torch.randn(1, 100, 2048)  # (batch, time, features)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([1, 100, 512])
    """

    def __init__(self, feature_dim=2048, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim

    def forward(self, x, mask=None):
        """Process frame features independently.

        Args:
            x (torch.Tensor): Input features, shape (B, T, feature_dim).
            mask (torch.BoolTensor, optional): Valid frame mask, shape (B, T).
                Not used in baseline but included for API consistency.

        Returns:
            torch.Tensor: Hidden representations, shape (B, T, hidden_dim).
        """
        return self.fc(x)


class LSTMModel(nn.Module):
    """Bidirectional LSTM for temporal sequence modeling.

    Processes the full sequence of frame features through a multi-layer
    bidirectional LSTM, then projects the output to a common hidden dimension.
    Based on the SV-RCNet approach.

    Args:
        feature_dim (int): Input feature dimension. Default: 2048.
        hidden_dim (int): Output hidden dimension. Default: 512.
        num_layers (int): Number of LSTM layers. Default: 2.
        bidirectional (bool): Use bidirectional LSTM. Default: True.
        dropout (float): Dropout between LSTM layers. Default: 0.3.

    Example:
        >>> model = LSTMModel(feature_dim=2048, hidden_dim=512)
        >>> x = torch.randn(1, 100, 2048)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([1, 100, 512])
    """

    def __init__(self, feature_dim=2048, hidden_dim=512, num_layers=2,
                 bidirectional=True, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim

    def forward(self, x, mask=None):
        """Process frame features through bidirectional LSTM.

        Args:
            x (torch.Tensor): Input features, shape (B, T, feature_dim).
            mask (torch.BoolTensor, optional): Valid frame mask, shape (B, T).
                Used for packing padded sequences.

        Returns:
            torch.Tensor: Hidden representations, shape (B, T, hidden_dim).
        """
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            output, _ = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=x.shape[1]
            )
        else:
            output, _ = self.lstm(x)

        return self.projection(output)


class DilatedConvBlock(nn.Module):
    """Single dilated convolution block used within a TCN stage.

    Applies a 1D dilated causal convolution followed by ReLU and dropout.
    Includes a residual connection from input to output.

    Args:
        channels (int): Number of input and output channels.
        dilation (int): Dilation factor for the convolution.
        dropout (float): Dropout probability. Default: 0.3.

    Example:
        >>> block = DilatedConvBlock(channels=64, dilation=2)
        >>> x = torch.randn(1, 64, 100)  # (B, C, T)
        >>> out = block(x)
        >>> print(out.shape)
        torch.Size([1, 64, 100])
    """

    def __init__(self, channels, dilation, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Apply dilated convolution with residual connection.

        Args:
            x (torch.Tensor): Input tensor, shape (B, C, T).

        Returns:
            torch.Tensor: Output tensor, shape (B, C, T).
        """
        residual = x
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        return out + residual


class TCNStage(nn.Module):
    """One stage of the Multi-Stage TCN.

    Consists of an input projection (1x1 conv) followed by a stack of
    dilated convolution blocks with exponentially increasing dilation.

    Args:
        input_dim (int): Input channel dimension.
        channels (int): Number of channels within the stage.
        num_layers (int): Number of dilated convolution blocks.
        dropout (float): Dropout probability. Default: 0.3.

    Example:
        >>> stage = TCNStage(input_dim=2048, channels=64, num_layers=10)
        >>> x = torch.randn(1, 2048, 100)  # (B, input_dim, T)
        >>> out = stage(x)
        >>> print(out.shape)
        torch.Size([1, 64, 100])
    """

    def __init__(self, input_dim, channels, num_layers, dropout=0.3):
        super().__init__()
        self.input_conv = nn.Conv1d(input_dim, channels, kernel_size=1)
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(DilatedConvBlock(channels, dilation, dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Process input through the TCN stage.

        Args:
            x (torch.Tensor): Input tensor, shape (B, input_dim, T).

        Returns:
            torch.Tensor: Output tensor, shape (B, channels, T).
        """
        x = self.input_conv(x)
        return self.layers(x)


class MultiStageTCN(nn.Module):
    """Multi-Stage Temporal Convolutional Network (MS-TCN).

    Chains multiple TCN stages where each stage refines the predictions
    of the previous one. The first stage takes raw features; subsequent
    stages take the output of the prior stage. Inspired by the TeCNO
    architecture (Czempiel et al., 2020).

    Args:
        feature_dim (int): Input feature dimension. Default: 2048.
        hidden_dim (int): Final output hidden dimension. Default: 512.
        num_stages (int): Number of TCN stages. Default: 4.
        num_layers (int): Number of dilated conv layers per stage. Default: 10.
        channels (int): Channel width within each stage. Default: 64.
        dropout (float): Dropout probability. Default: 0.3.

    Example:
        >>> model = MultiStageTCN(feature_dim=2048, hidden_dim=512,
        ...                       num_stages=4, num_layers=10, channels=64)
        >>> x = torch.randn(1, 100, 2048)  # (B, T, feature_dim)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([1, 100, 512])
    """

    def __init__(self, feature_dim=2048, hidden_dim=512, num_stages=4,
                 num_layers=10, channels=64, dropout=0.3):
        super().__init__()
        self.stages = nn.ModuleList()
        # First stage takes raw features
        self.stages.append(TCNStage(feature_dim, channels, num_layers, dropout))
        # Subsequent stages refine previous stage output
        for _ in range(num_stages - 1):
            self.stages.append(TCNStage(channels, channels, num_layers, dropout))

        self.projection = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim

    def forward(self, x, mask=None):
        """Process features through all TCN stages.

        Args:
            x (torch.Tensor): Input features, shape (B, T, feature_dim).
            mask (torch.BoolTensor, optional): Valid frame mask, shape (B, T).
                If provided, masked positions are zeroed after each stage.

        Returns:
            torch.Tensor: Hidden representations, shape (B, T, hidden_dim).
        """
        # TCN expects (B, C, T) format
        out = x.permute(0, 2, 1)

        for stage in self.stages:
            out = stage(out)
            if mask is not None:
                out = out * mask.unsqueeze(1).float()

        # Back to (B, T, C) and project to hidden_dim
        out = out.permute(0, 2, 1)
        return self.projection(out)

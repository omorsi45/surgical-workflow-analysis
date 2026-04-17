"""
ResNet-50 backbone for spatial feature extraction.

Uses a pretrained ResNet-50 from torchvision with the final classification
layer removed. Outputs a 2048-dimensional feature vector per input frame.
Features are extracted once and cached to disk.

Author: Omar Morsi (40236376)
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm


class ResNet50FeatureExtractor(nn.Module):
    """Pretrained ResNet-50 with final FC layer removed for feature extraction.

    Loads ResNet-50 weights pretrained on ImageNet and uses all layers up to
    (but not including) the final fully-connected classification layer.
    The output is a 2048-dim feature vector per frame.

    Args:
        pretrained (bool): Whether to load ImageNet pretrained weights.
            Default: True.

    Attributes:
        backbone (nn.Sequential): ResNet-50 layers up to average pooling.
        feature_dim (int): Output feature dimension (2048).

    Example:
        >>> extractor = ResNet50FeatureExtractor(pretrained=True)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> features = extractor(x)
        >>> print(features.shape)
        torch.Size([4, 2048])
    """

    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None
        resnet = models.resnet50(weights=weights)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Extract spatial features from input frames.

        Args:
            x (torch.Tensor): Batch of images, shape (B, 3, 224, 224).

        Returns:
            torch.Tensor: Feature vectors, shape (B, 2048).
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features.squeeze(-1).squeeze(-1)

    @torch.no_grad()
    def extract_and_cache(self, dataset, save_path, video_id, batch_size=32,
                          device="cuda"):
        """Extract features from an entire video dataset and save to disk.

        Processes the dataset in batches, extracts 2048-dim features for each
        frame, and saves features along with phase and tool labels as a
        single .pt file.

        Args:
            dataset (Cholec80VideoDataset): Video frame dataset for one video.
            save_path (str): Directory to save the .pt feature file.
            video_id (int): Video number, used for the output filename.
            batch_size (int): Batch size for feature extraction. Default: 32.
            device (str): Device to run extraction on. Default: "cuda".

        Returns:
            None. Saves a .pt file at save_path/video{video_id:02d}.pt
            containing keys "features", "phases", "tools".

        Example:
            >>> extractor = ResNet50FeatureExtractor()
            >>> dataset = Cholec80VideoDataset("data/cholec80", video_id=1,
            ...                                transform=get_eval_transforms())
            >>> extractor.extract_and_cache(dataset, "data/features",
            ...                             video_id=1, device="cuda")
        """
        self.eval()
        self.to(device)
        # Workers > 0 require CUDA context to be initialized first; fall back
        # to 0 on CPU-only machines to avoid multiprocessing overhead.
        nw = 0 if (device == "cpu" or not torch.cuda.is_available()) else 2
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=nw
        )

        all_features = []
        all_phases = []
        all_tools = []

        for images, phases, tools in tqdm(
            loader, desc=f"Extracting video{video_id:02d}"
        ):
            images = images.to(device)
            features = self.forward(images)
            all_features.append(features.cpu())
            all_phases.append(phases)
            all_tools.append(tools)

        all_features = torch.cat(all_features, dim=0)
        all_phases = torch.cat(all_phases, dim=0)
        all_tools = torch.cat(all_tools, dim=0)

        os.makedirs(save_path, exist_ok=True)
        torch.save(
            {"features": all_features, "phases": all_phases, "tools": all_tools},
            os.path.join(save_path, f"video{video_id:02d}.pt"),
        )
        print(f"  Saved {all_features.shape[0]} frames -> {save_path}/video{video_id:02d}.pt")

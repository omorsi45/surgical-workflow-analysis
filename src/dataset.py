"""
Cholec80 dataset loading, preprocessing, and feature caching.

Handles reading video frames and annotation files from the Cholec80 dataset,
subsampling to 1 fps, applying image augmentations, and caching ResNet-50
features to disk for efficient temporal model training.

Author: Omar Morsi (40236376)
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2


PHASE_NAMES = [
    "Preparation",
    "Calot Triangle Dissection",
    "Clipping and Cutting",
    "Gallbladder Dissection",
    "Gallbladder Packaging",
    "Cleaning and Coagulation",
    "Gallbladder Retraction",
]

# Cholec80 annotation files use compact names without spaces.
# Map both compact and spaced variants to phase indices.
PHASE_NAME_TO_IDX = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "Calot Triangle Dissection": 1,
    "ClippingCutting": 2,
    "Clipping and Cutting": 2,
    "GallbladderDissection": 3,
    "Gallbladder Dissection": 3,
    "GallbladderPackaging": 4,
    "Gallbladder Packaging": 4,
    "CleaningCoagulation": 5,
    "Cleaning and Coagulation": 5,
    "GallbladderRetraction": 6,
    "Gallbladder Retraction": 6,
}

TOOL_NAMES = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag",
]


def parse_phase_annotations(annotation_path):
    """Parse a Cholec80 phase annotation file.

    Supports two annotation formats:
    - Integer indices: ``'<frame_number>\\t<phase_index>'``
    - Text names: ``'<frame_number>\\t<PhaseName>'`` (e.g. ``CalotTriangleDissection``)

    The first line is a header and is skipped. Frame numbers use the original
    25-fps numbering (0, 1, 2, …), so callers that want 1-fps data should
    filter to every 25th frame.

    Args:
        annotation_path (str): Path to the phase annotation .txt file.

    Returns:
        dict: Mapping from frame number (int) to phase index (int, 0-6).

    Example:
        >>> phases = parse_phase_annotations("data/phase_annotations/video01-phase.txt")
        >>> print(phases[0])
        0
    """
    phase_map = {}
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:
        # Cholec80 annotations may use tabs or spaces as delimiters
        parts = line.strip().split()
        if len(parts) >= 2:
            frame_num = int(parts[0])
            label = parts[1]
            if label.lstrip("-").isdigit():
                phase_idx = int(label)
            else:
                phase_idx = PHASE_NAME_TO_IDX.get(label, 0)
            phase_map[frame_num] = phase_idx
    return phase_map


def parse_tool_annotations(annotation_path):
    """Parse a Cholec80 tool annotation file.

    Each line has the format: '<frame_number>\\t<tool1>\\t<tool2>\\t...\\t<tool7>'
    where each tool value is 0 or 1. The first line is a header.

    Args:
        annotation_path (str): Path to the tool annotation .txt file.

    Returns:
        dict: Mapping from frame number (int) to a numpy array of shape (7,)
            with binary tool presence indicators.

    Example:
        >>> tools = parse_tool_annotations("data/cholec80/video01-tool.txt")
        >>> print(tools[0])
        array([1, 0, 0, 0, 0, 0, 0])
    """
    tool_map = {}
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:
        # Cholec80 annotations may use tabs or spaces as delimiters
        parts = line.strip().split()
        if len(parts) >= 8:
            frame_num = int(parts[0])
            tool_vec = np.array([int(x) for x in parts[1:8]], dtype=np.float32)
            tool_map[frame_num] = tool_vec
    return tool_map


def get_train_transforms(frame_size=224):
    """Build the training image transform pipeline.

    Applies random augmentations (flip, color jitter, rotation, blur)
    followed by resize, tensor conversion, and ImageNet normalization.

    Args:
        frame_size (int): Target spatial size for resizing. Default: 224.

    Returns:
        torchvision.transforms.Compose: The composed transform pipeline.

    Example:
        >>> transform = get_train_transforms(224)
        >>> from PIL import Image
        >>> img = Image.new("RGB", (480, 854))
        >>> tensor = transform(img)
        >>> print(tensor.shape)
        torch.Size([3, 224, 224])
    """
    return T.Compose([
        T.ToPILImage(),
        T.Resize((frame_size, frame_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomRotation(degrees=10),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(frame_size=224):
    """Build the validation/test image transform pipeline.

    Applies only resize, tensor conversion, and ImageNet normalization
    (no augmentation).

    Args:
        frame_size (int): Target spatial size for resizing. Default: 224.

    Returns:
        torchvision.transforms.Compose: The composed transform pipeline.

    Example:
        >>> transform = get_eval_transforms(224)
        >>> from PIL import Image
        >>> img = Image.new("RGB", (480, 854))
        >>> tensor = transform(img)
        >>> print(tensor.shape)
        torch.Size([3, 224, 224])
    """
    return T.Compose([
        T.ToPILImage(),
        T.Resize((frame_size, frame_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class Cholec80VideoDataset(Dataset):
    """Dataset for loading raw Cholec80 video frames with annotations.

    Used during the feature extraction phase to feed frames through ResNet-50.
    Loads frames from a single video at 1 fps and returns image tensors with
    corresponding phase and tool labels.

    Args:
        data_dir (str): Root directory of the Cholec80 dataset.
        video_id (int): Video number (1-80).
        fps (int): Frames per second to subsample to. Default: 1.
        transform (callable, optional): Transform to apply to each frame.

    Attributes:
        frames (list): List of (frame_path, frame_number) tuples.
        phase_labels (dict): Frame number to phase index mapping.
        tool_labels (dict): Frame number to tool vector mapping.

    Example:
        >>> dataset = Cholec80VideoDataset("data/cholec80", video_id=1,
        ...                                transform=get_eval_transforms())
        >>> img, phase, tools = dataset[0]
        >>> print(img.shape, phase, tools.shape)
        torch.Size([3, 224, 224]) 0 torch.Size([7])
    """

    def __init__(self, data_dir, video_id, fps=1, transform=None):
        self.data_dir = data_dir
        self.video_id = video_id
        self.fps = fps
        self.transform = transform

        video_name = f"video{video_id:02d}"
        video_dir = os.path.join(data_dir, video_name)

        # ── Locate annotation files ──────────────────────────────────────
        # Search order (most-to-least specific):
        #   1. data_dir/phase_annotations/videoXX-phase.txt  (downloaded layout)
        #   2. data_dir/videoXX-phase.txt                    (flat cholec80 layout)
        #   3. data_dir/videoXX/videoXX-phase.txt            (per-video folder)
        _phase_candidates = [
            os.path.join(data_dir, "phase_annotations", f"{video_name}-phase.txt"),
            os.path.join(data_dir, f"{video_name}-phase.txt"),
            os.path.join(video_dir, f"{video_name}-phase.txt"),
        ]
        _tool_candidates = [
            os.path.join(data_dir, "tool_annotations", f"{video_name}-tool.txt"),
            os.path.join(data_dir, f"{video_name}-tool.txt"),
            os.path.join(video_dir, f"{video_name}-tool.txt"),
        ]
        phase_path = next((p for p in _phase_candidates if os.path.exists(p)), _phase_candidates[0])
        tool_path = next((p for p in _tool_candidates if os.path.exists(p)), _tool_candidates[0])

        self.phase_labels = parse_phase_annotations(phase_path)
        self.tool_labels = parse_tool_annotations(tool_path)

        # ── Locate video frames ──────────────────────────────────────────
        # Prefer pre-extracted PNG/JPG frames in a per-video folder; fall
        # back to reading directly from an MP4 file.
        #
        # Frame numbers follow the original 25-fps numbering.  Tool
        # annotations are already at 1 fps (every 25th frame: 0, 25, 50 …),
        # so we use their keys as the canonical 1-fps frame list and look up
        # the matching phase label at the same frame number.
        all_frames = sorted(glob.glob(os.path.join(video_dir, "*.png")))
        if not all_frames:
            all_frames = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))

        self.mp4_path = None
        self._frame_cache = {}
        if all_frames:
            # Pre-extracted frames: keep only those with annotation entries.
            self.frames = []
            for frame_path in all_frames:
                fname = os.path.basename(frame_path)
                frame_num = int(os.path.splitext(fname)[0])
                if frame_num in self.phase_labels:
                    self.frames.append((frame_path, frame_num))
        else:
            # MP4 fall-back: search common locations.
            _mp4_candidates = [
                os.path.join(data_dir, "videos", f"{video_name}.mp4"),
                os.path.join(data_dir, f"{video_name}.mp4"),
            ]
            self.mp4_path = next(
                (p for p in _mp4_candidates if os.path.exists(p)), None
            )
            # Use the tool annotation keys as the 1-fps frame list (they are
            # already spaced every 25 frames in 25-fps numbering).
            frame_nums = sorted(
                fn for fn in self.tool_labels if fn in self.phase_labels
            )
            self.frames = [(None, fn) for fn in frame_nums]
            # Pre-read all frames once to avoid re-opening VideoCapture per __getitem__
            self._frame_cache = {}
            if self.mp4_path is not None:
                cap = cv2.VideoCapture(self.mp4_path)
                for fn in frame_nums:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                    ret, img = cap.read()
                    self._frame_cache[fn] = img if ret else None
                cap.release()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_path, frame_num = self.frames[idx]

        if self.mp4_path is not None:
            img = self._frame_cache.get(frame_num)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.imread(frame_path)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        phase = self.phase_labels.get(frame_num, 0)
        tools = self.tool_labels.get(frame_num, np.zeros(7, dtype=np.float32))

        return img, torch.tensor(phase, dtype=torch.long), torch.tensor(tools, dtype=torch.float32)


class Cholec80FeatureDataset(Dataset):
    """Dataset for loading pre-extracted ResNet-50 features for temporal modeling.

    Loads cached feature vectors for an entire video as a sequence, along with
    the corresponding phase and tool labels. Each item is a full video sequence
    used for training temporal models.

    Args:
        features_dir (str): Directory containing cached .pt feature files.
        video_ids (list): List of video IDs (ints) to include.

    Example:
        >>> dataset = Cholec80FeatureDataset("data/features", [1, 2, 3])
        >>> features, phases, tools = dataset[0]
        >>> print(features.shape)
        torch.Size([1200, 2048])
    """

    def __init__(self, features_dir, video_ids):
        self.features_dir = features_dir
        self.video_ids = video_ids

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        data = torch.load(
            os.path.join(self.features_dir, f"video{video_id:02d}.pt"),
            weights_only=True,
        )
        features = data["features"]
        phases = data["phases"]
        tools = data["tools"]
        return features, phases, tools


def collate_sequences(batch):
    """Custom collate function for variable-length video sequences.

    Pads all sequences in the batch to the length of the longest sequence.
    Creates a mask tensor to indicate valid (non-padded) time steps.

    Args:
        batch (list): List of (features, phases, tools) tuples from
            Cholec80FeatureDataset.

    Returns:
        tuple: (features, phases, tools, mask) where:
            - features: Tensor of shape (B, T_max, 2048)
            - phases: Tensor of shape (B, T_max), padded with -1
            - tools: Tensor of shape (B, T_max, 7), padded with 0
            - mask: BoolTensor of shape (B, T_max), True for valid steps

    Example:
        >>> batch = [(torch.randn(100, 2048), torch.zeros(100), torch.zeros(100, 7)),
        ...          (torch.randn(150, 2048), torch.zeros(150), torch.zeros(150, 7))]
        >>> feats, phases, tools, mask = collate_sequences(batch)
        >>> print(feats.shape, mask.shape)
        torch.Size([2, 150, 2048]) torch.Size([2, 150])
    """
    features_list, phases_list, tools_list = zip(*batch)
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    batch_size = len(batch)
    feat_dim = features_list[0].shape[1]

    features = torch.zeros(batch_size, max_len, feat_dim)
    phases = torch.full((batch_size, max_len), -1, dtype=torch.long)
    tools = torch.zeros(batch_size, max_len, 7)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, length in enumerate(lengths):
        features[i, :length] = features_list[i]
        phases[i, :length] = phases_list[i]
        tools[i, :length] = tools_list[i]
        mask[i, :length] = True

    return features, phases, tools, mask

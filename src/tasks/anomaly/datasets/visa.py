import os
import numpy as np
import torch
from PIL import Image

from src.core.errors import LocalAssetError
from src.core.registry import DATASETS
from src.data.split import assert_disjoint, load_split_file
from .base import BaseAnomalyDataset

TRAIN_PREFIX = "train_normal"
SUPPORTED_EXTENSIONS = [".JPG", ".jpg", ".png", ".PNG", ".jpeg", ".JPEG", ".bmp", ".BMP"]


def _find_file(directory, stem):
    for ext in SUPPORTED_EXTENSIONS:
        candidate = os.path.join(directory, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"File for stem '{stem}' not found under '{directory}'")


@DATASETS.register("visa_anomaly")
class ViSAAnomaly(BaseAnomalyDataset):
    """VisA (Visual Anomaly) dataset."""

    def __init__(self, root, split, transform=None, split_path=None, category="candle", **params):
        super().__init__(root, split, transform=transform, **params)
        if split not in ("train", "valid", "test"):
            raise ValueError(f"split must be one of train/valid/test, got '{split}'")
        if split_path is None:
            raise LocalAssetError(
                "ViSAAnomaly requires a split file path (data.split.path in config)."
            )

        category_dir = os.path.join(root, category)
        image_normal_dir = os.path.join(category_dir, "Data", "Images", "Normal")
        image_anomaly_dir = os.path.join(category_dir, "Data", "Images", "Anomaly")
        mask_anomaly_dir = os.path.join(category_dir, "Data", "Masks", "Anomaly")

        if not os.path.isdir(category_dir) or not os.path.isdir(image_normal_dir):
            raise LocalAssetError(
                f"VisA category '{category}' not found under root='{root}'. "
                f"Expected '{image_normal_dir}' to exist."
            )
        if not os.path.isfile(split_path):
            raise LocalAssetError(f"Split file not found: {split_path}")

        self.category = category
        self.category_dir = category_dir
        self.image_normal_dir = image_normal_dir
        self.image_anomaly_dir = image_anomaly_dir
        self.mask_anomaly_dir = mask_anomaly_dir
        self.split_path = split_path

        split_dict = load_split_file(split_path)
        assert_disjoint(split_dict)
        self.ids = sorted(split_dict[split])

    def __len__(self):
        return len(self.ids)

    def _image_path(self, type_prefix, stem):
        if type_prefix in (TRAIN_PREFIX, "Normal"):
            return _find_file(self.image_normal_dir, stem)
        return _find_file(self.image_anomaly_dir, stem)

    def _mask_path(self, stem):
        return _find_file(self.mask_anomaly_dir, stem)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        type_prefix, stem = sample_id.split("/", 1)
        image_path = self._image_path(type_prefix, stem)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image '{image_path}' (sample_id='{sample_id}'): {exc}"
            ) from exc

        if self.split == "train":
            target = {}
            if self.transform is not None:
                return self.transform(image, target)
            import torchvision.transforms.v2.functional as F
            return F.to_dtype(F.to_image(image), torch.float32, scale=True), target

        is_anomalous = (type_prefix == "Anomaly")
        width, height = image.size

        if is_anomalous:
            mask_path = self._mask_path(stem)
            try:
                mask_array = np.array(Image.open(mask_path).convert("L"))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load mask '{mask_path}' (sample_id='{sample_id}'): {exc}"
                ) from exc
            mask_array = (mask_array > 0).astype(np.int64)
        else:
            mask_array = np.zeros((height, width), dtype=np.int64)

        mask = torch.from_numpy(mask_array).to(torch.int64)
        label = torch.tensor(int(is_anomalous), dtype=torch.long)
        target = {"label": label, "mask": mask}

        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            import torchvision.transforms.v2.functional as F
            image = F.to_dtype(F.to_image(image), torch.float32, scale=True)

        return image, target

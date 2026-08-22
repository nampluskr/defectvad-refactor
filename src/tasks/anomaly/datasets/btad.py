import os
import numpy as np
import torch
from PIL import Image

from src.core.errors import LocalAssetError
from src.core.registry import DATASETS
from src.data.split import assert_disjoint, generate_ratio_split, load_split_file
from .base import BaseAnomalyDataset

TRAIN_PREFIX = "train_ok"
BTAD_CATEGORIES = ["01", "02", "03"]
SUPPORTED_EXTENSIONS = [".bmp", ".png", ".jpg", ".jpeg", ".BMP", ".PNG", ".JPG"]


def _find_file(directory, stem):
    for ext in SUPPORTED_EXTENSIONS:
        candidate = os.path.join(directory, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"File for stem '{stem}' not found under '{directory}'")


@DATASETS.register("btad_anomaly")
class BTADAnomaly(BaseAnomalyDataset):
    """BTAD (BeanTech Anomaly Detection) dataset."""

    def __init__(self, root, split, transform=None, split_path=None, category="01", **params):
        super().__init__(root, split, transform=transform, **params)
        if split not in ("train", "valid", "test"):
            raise ValueError(f"split must be one of train/valid/test, got '{split}'")
        if split_path is None:
            raise LocalAssetError(
                "BTADAnomaly requires a split file path (data.split.path in config)."
            )

        category_str = str(category).zfill(2)
        category_dir = os.path.join(root, category_str)
        train_dir = os.path.join(category_dir, "train", "ok")
        test_dir = os.path.join(category_dir, "test")
        ground_truth_dir = os.path.join(category_dir, "ground_truth", "ko")

        if not os.path.isdir(category_dir) or not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
            raise LocalAssetError(
                f"BTAD category '{category_str}' not found under root='{root}'. "
                f"Expected '{train_dir}' and '{test_dir}' to exist."
            )
        if not os.path.isfile(split_path):
            raise LocalAssetError(f"Split file not found: {split_path}")

        self.category = category_str
        self.category_dir = category_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.ground_truth_dir = ground_truth_dir
        self.split_path = split_path

        split_dict = load_split_file(split_path)
        assert_disjoint(split_dict)
        self.ids = sorted(split_dict[split])

    def __len__(self):
        return len(self.ids)

    def _image_path(self, defect_type, stem):
        if defect_type == TRAIN_PREFIX:
            return _find_file(self.train_dir, stem)
        return _find_file(os.path.join(self.test_dir, defect_type), stem)

    def _mask_path(self, stem):
        return _find_file(self.ground_truth_dir, stem)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        defect_type, stem = sample_id.split("/", 1)
        image_path = self._image_path(defect_type, stem)

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

        is_anomalous = (defect_type == "ko")
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

    @classmethod
    def generate_split(cls, dataset_root, category, seed=42, ratio=(0.0, 0.4, 0.6)):
        category_str = str(category).zfill(2)
        category_dir = os.path.join(dataset_root, category_str)
        train_dir = os.path.join(category_dir, "train", "ok")
        test_ok_dir = os.path.join(category_dir, "test", "ok")
        test_ko_dir = os.path.join(category_dir, "test", "ko")

        if not os.path.isdir(category_dir) or not os.path.isdir(train_dir):
            raise LocalAssetError(
                f"BTAD category '{category_str}' not found under '{dataset_root}'."
            )

        valid_exts = {ext.lower() for ext in SUPPORTED_EXTENSIONS}
        train_files = sorted(os.listdir(train_dir)) if os.path.isdir(train_dir) else []
        train_ids = [
            f"{TRAIN_PREFIX}/{os.path.splitext(name)[0]}"
            for name in train_files
            if os.path.splitext(name)[1].lower() in valid_exts
        ]

        test_ids = []
        defect_types = []

        if os.path.isdir(test_ok_dir):
            for name in sorted(os.listdir(test_ok_dir)):
                if os.path.splitext(name)[1].lower() in valid_exts:
                    test_ids.append(f"ok/{os.path.splitext(name)[0]}")
                    defect_types.append("ok")

        if os.path.isdir(test_ko_dir):
            for name in sorted(os.listdir(test_ko_dir)):
                if os.path.splitext(name)[1].lower() in valid_exts:
                    test_ids.append(f"ko/{os.path.splitext(name)[0]}")
                    defect_types.append("ko")

        ratio_split = generate_ratio_split(
            test_ids,
            ratio={"train": ratio[0], "valid": ratio[1], "test": ratio[2]},
            seed=seed,
            stratify_by=defect_types,
        )

        return {
            "train": sorted(train_ids),
            "valid": sorted(ratio_split["valid"]),
            "test": sorted(ratio_split["test"]),
        }

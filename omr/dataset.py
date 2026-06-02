from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from omr.utils import read_jsonl


def _get_transforms_module():
    from torchvision import transforms
    return transforms


class OMRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        img_size: int = 64,
        augment: bool = False,
        mean: float = 0.5,
        std: float = 0.5,
    ) -> None:
        self.items = read_jsonl(manifest_path)
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.augment = augment
        self.transform = self._build_transform()

    def _build_transform(self):
        transforms = _get_transforms_module()
        ops: list[Any] = [transforms.Grayscale(num_output_channels=1)]
        if self.augment:
            ops.extend(
                [
                    transforms.RandomAffine(
                        degrees=8,
                        translate=(0.06, 0.06),
                        scale=(0.92, 1.08),
                        shear=5,
                        fill=255,
                    ),
                    transforms.RandomPerspective(distortion_scale=0.18, p=0.25, fill=255),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                ]
            )
        ops.extend(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mean], std=[self.std]),
            ]
        )
        if self.augment:
            ops.append(transforms.RandomErasing(p=0.10, scale=(0.02, 0.10), value=1.0))
        return transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image = Image.open(item["image"]).convert("L")
        tensor = self.transform(image)
        label = float(item["label"])
        return {
            "image": tensor,
            "label": torch.tensor([label], dtype=torch.float32),
            "path": item["image"],
            "meta": item,
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.cat([b["label"] for b in batch], dim=0).unsqueeze(1)
    return {
        "image": images,
        "label": labels,
        "path": [b["path"] for b in batch],
        "meta": [b["meta"] for b in batch],
    }


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    img_size = int(model_cfg.get("img_size", 64))

    train_ds = OMRDataset(
        data_cfg["train_manifest"],
        img_size=img_size,
        augment=True,
        mean=float(model_cfg.get("mean", 0.5)),
        std=float(model_cfg.get("std", 0.5)),
    )
    val_ds = OMRDataset(
        data_cfg["val_manifest"],
        img_size=img_size,
        augment=False,
        mean=float(model_cfg.get("mean", 0.5)),
        std=float(model_cfg.get("std", 0.5)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 128)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=True,
        collate_fn=_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=True,
        collate_fn=_collate,
        drop_last=False,
    )

    stats = {
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "train_pos": sum(int(float(x["label"]) > 0.5) for x in train_ds.items),
        "val_pos": sum(int(float(x["label"]) > 0.5) for x in val_ds.items),
    }
    return train_loader, val_loader, stats


def preprocess_crop_for_inference(
    image: Image.Image,
    img_size: int = 64,
    mean: float = 0.5,
    std: float = 0.5,
) -> torch.Tensor:
    transforms = _get_transforms_module()
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )
    return transform(image)


__all__ = ["OMRDataset", "build_dataloaders", "preprocess_crop_for_inference"]

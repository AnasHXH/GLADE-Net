import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .augment import AugmentRGBTorch


def load_img(filepath: str) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


class ImagePairDataset(Dataset):
    def __init__(self, root_dir: str, img_size: int = 128, split: str = "train"):
        assert split in {"train", "val"}
        gt_dir = os.path.join(root_dir, "label")
        in_dir = os.path.join(root_dir, "cloud")
        self.clean_files = sorted(os.listdir(gt_dir))
        self.noisy_files = sorted(os.listdir(in_dir))
        self.clean_paths = [os.path.join(gt_dir, x) for x in self.clean_files]
        self.noisy_paths = [os.path.join(in_dir, x) for x in self.noisy_files]
        self.img_size = img_size
        self.augment = AugmentRGBTorch()

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean = torch.from_numpy(load_img(self.clean_paths[idx]))
        noisy = torch.from_numpy(load_img(self.noisy_paths[idx]))
        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        C, H, W = clean.shape
        ps = self.img_size
        r = 0 if H - ps == 0 else np.random.randint(0, H - ps)
        c = 0 if W - ps == 0 else np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        # same random aug
        aug_id = np.random.randint(0, 8)
        clean = self.augment.methods[aug_id](clean)
        noisy = self.augment.methods[aug_id](noisy)
        return clean, noisy
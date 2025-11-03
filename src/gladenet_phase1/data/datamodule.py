from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .datasets import ImagePairDataset


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, train_dir: str, val_dir: str, img_size: int, batch_size: int, num_workers: int):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = ImagePairDataset(self.train_dir, img_size=self.img_size, split="train")
        self.val_ds = ImagePairDataset(self.val_dir, img_size=self.img_size, split="val")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
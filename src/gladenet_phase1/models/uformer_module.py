import torch
import torch.optim as optim
import pytorch_lightning as pl

from ..losses.gradual_frequency_loss import GradualFrequencyLoss
from ..utils.metrics import psnr
from .safe_imports import UformerModel


class UFormerModule(pl.LightningModule):
    def __init__(self, img_size: int = 128, lr: float = 2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.net = UformerModel(
            img_size=img_size,
            embed_dim=16,
            win_size=8,
            token_projection="linear",
            token_mlp="leff",
        )
        self.crit = GradualFrequencyLoss(start_frequency=10, end_frequency=128, num_epochs=64, apply_lap=False)
        self.best_train_psnr = []
        self.best_val_psnr = []

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.hparams.lr)
        sch = optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.5)
        return {"optimizer": opt, "lr_scheduler": sch}

    def training_step(self, batch, batch_idx):
        clean, hazy = batch
        pred = torch.clamp(self(hazy), 0, 1)
        loss = self.crit(pred, clean, self.current_epoch)
        train_psnr = psnr(pred, clean)
        self.best_train_psnr.append(train_psnr)
        self.log_dict({"train_loss": loss, "train_psnr": train_psnr}, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        clean, hazy = batch
        pred = torch.clamp(self(hazy), 0, 1)
        loss = self.crit(pred, clean, self.current_epoch)
        val_psnr = psnr(pred, clean)
        self.best_val_psnr.append(val_psnr)
        self.log_dict({"valid_loss": loss, "valid_psnr": val_psnr}, prog_bar=True, on_epoch=True)
        return loss
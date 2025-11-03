import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.gladenet_phase1.data.datamodule import ImageDataModule
from src.gladenet_phase1.models.uformer_module import UFormerModule
from src.gladenet_phase1.utils.seed import seed_all



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', type=str, required=True)
    p.add_argument('--val_dir', type=str, required=True)
    p.add_argument('--img_size', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=2500)
    p.add_argument('--devices', type=int, default=1)
    p.add_argument('--precision', type=int, default=16)
    p.add_argument('--accumulate', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--save_ckpt', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    dm = ImageDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=os.cpu_count() or 8,
    )
    model = UFormerModule(img_size=args.img_size, lr=args.lr)

    ckpt_dir = os.path.join('lightning_logs', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir, save_top_k=2, monitor='valid_psnr', mode='max', filename='{epoch}-{valid_psnr:.3f}'),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=args.precision,
        accelerator='gpu' if args.devices > 0 else 'cpu',
        devices=args.devices,
        accumulate_grad_batches=args.accumulate,
        callbacks=callbacks,
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)

    if args.save_ckpt:
        trainer.save_checkpoint(args.save_ckpt)


if __name__ == '__main__':
    main()
import pytorch_lightning as pl

def seed_all(seed: int = 42):
    pl.seed_everything(seed, workers=True)
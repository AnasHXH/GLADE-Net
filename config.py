from dataclasses import dataclass

@dataclass
class TrainConfig:
    img_size: int = 128
    warmup_epochs: int = 10
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 64
    num_workers: int = 16
import random
import torch

class AugmentRGBTorch:
    def __init__(self):
        self.methods = [
            self.transform0,
            self.transform1,
            self.transform2,
            self.transform3,
            self.transform4,
            self.transform5,
            self.transform6,
            self.transform7,
        ]

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        fn = random.choice(self.methods)
        return fn(t)

    def transform0(self, t):
        return t

    def transform1(self, t):
        return torch.rot90(t, k=1, dims=[-1, -2])

    def transform2(self, t):
        return torch.rot90(t, k=2, dims=[-1, -2])

    def transform3(self, t):
        return torch.rot90(t, k=3, dims=[-1, -2])

    def transform4(self, t):
        return t.flip(-2)

    def transform5(self, t):
        return torch.rot90(t, k=1, dims=[-1, -2]).flip(-2)

    def transform6(self, t):
        return torch.rot90(t, k=2, dims=[-1, -2]).flip(-2)

    def transform7(self, t):
        return torch.rot90(t, k=3, dims=[-1, -2]).flip(-2)
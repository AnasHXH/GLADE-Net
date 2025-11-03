import torch
import torch.nn as nn
import torch.nn.functional as F

class _LaplacianKernel(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[.05, .25, .4, .25, .05]], dtype=torch.float32)
        kernel = (k.T @ k).unsqueeze(0).unsqueeze(0)  # 1x1x5x5
        self.register_buffer("kernel", kernel)

    def conv_gauss(self, img):
        n, c, h, w = img.shape
        k = self.kernel.to(img.device)
        k = k.expand(c, 1, 5, 5)
        img = F.pad(img, (2, 2, 2, 2), mode='replicate')
        return F.conv2d(img, k, groups=c)

    def forward(self, x):
        # simple Laplacian approx via do-down-up-do
        filtered = self.conv_gauss(x)
        down = filtered[:, :, ::2, ::2]
        up = torch.zeros_like(filtered)
        up[:, :, ::2, ::2] = down * 4
        filtered2 = self.conv_gauss(up)
        return x - filtered2


class GradualFrequencyLoss(nn.Module):
    """
    Band-pass style loss that expands from ultra-high frequencies toward lower frequencies over epochs.
    """
    def __init__(self, start_frequency=10, end_frequency=128, num_epochs=64, apply_lap=False, eps: float = 1e-3):
        super().__init__()
        self.start_frequency = start_frequency
        self.end_frequency = end_frequency
        self.num_epochs = num_epochs
        self.apply_lap = apply_lap
        self.eps = eps
        self.lap = _LaplacianKernel()

    def forward(self, pred, target, current_epoch: int):
        # compute cut-off that moves downward in frequency as epochs increase
        cut = self.start_frequency - (self.start_frequency - self.end_frequency) * current_epoch // max(1, self.num_epochs)

        def _fft_mask(x):
            b, c, h, w = x.shape
            xfft = torch.fft.fft2(x, norm='ortho')
            # rectangular mask: keep rows < cut and cols < cut
            mask = torch.zeros_like(xfft.real)
            mask[:, :, :cut, :] = 1
            mask[:, :, :, :cut] = 1
            x_re = torch.stack([xfft.real * mask, xfft.imag * mask], dim=2)  # (B,C,2,H,W)
            if self.apply_lap:
                x_mag = torch.fft.ifft2(torch.complex(x_re[:, :, 0], x_re[:, :, 1]), norm='ortho').real
                x_mag = self.lap(x_mag)
                return x_mag
            else:
                return x_re

        fp = _fft_mask(pred)
        ft = _fft_mask(target)
        diff = fp - ft
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
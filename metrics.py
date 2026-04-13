"""
metrics.py — PSNR, SSIM, and FID helpers for WG-DM evaluation.
"""

import torch
import torch.nn.functional as F
import math


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """PSNR in dB. Both tensors in [0, 1], shape (B, C, H, W)."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(max_val) - 10 * math.log10(mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                 window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """
    Structural Similarity Index. Simple single-scale implementation.
    Both tensors in [0, 1], shape (B, C, H, W).
    """
    channel = pred.shape[1]
    # Gaussian kernel
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    kernel = g[:, None] * g[None, :]
    kernel = kernel.expand(channel, 1, window_size, window_size).to(pred.device)

    pad = window_size // 2
    mu1 = F.conv2d(pred,   kernel, padding=pad, groups=channel)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=channel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=channel) - mu2_sq
    sigma12   = F.conv2d(pred * target, kernel, padding=pad, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


# ─────────────────────────────────────────────
# FID (uses torchmetrics.image.fid if available)
# ─────────────────────────────────────────────

def compute_fid(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    """
    Wrapper around torchmetrics FID. Requires:
        pip install torchmetrics[image]
    Both tensors: (N, 3, H, W) in [0, 1], H/W >= 299 for InceptionV3.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("Install torchmetrics[image] for FID: pip install 'torchmetrics[image]'")
        return float("nan")

    fid = FrechetInceptionDistance(normalize=True).to(real_images.device)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute().item()

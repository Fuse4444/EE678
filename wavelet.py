"""
wavelet.py — 2D DWT / IDWT utilities for WG-DM
Uses PyWavelets (pywt) and operates on torch tensors.
"""

import torch
import torch.nn as nn
import pywt
import numpy as np


def dwt2d(x: torch.Tensor, wavelet: str = "haar") -> dict:
    """
    Apply a single-level 2D DWT to a batch of images.

    Args:
        x: (B, C, H, W) float tensor in [-1, 1]
        wavelet: pywt wavelet name (haar, db2, bior2.2, etc.)

    Returns:
        dict with keys 'LL', 'LH', 'HL', 'HH', each (B, C, H//2, W//2)
    """
    B, C, H, W = x.shape
    device = x.device
    x_np = x.detach().cpu().numpy()  # (B, C, H, W)

    ll_list, lh_list, hl_list, hh_list = [], [], [], []
    for b in range(B):
        ch_ll, ch_lh, ch_hl, ch_hh = [], [], [], []
        for c in range(C):
            coeffs2 = pywt.dwt2(x_np[b, c], wavelet)
            LL, (LH, HL, HH) = coeffs2
            ch_ll.append(LL)
            ch_lh.append(LH)
            ch_hl.append(HL)
            ch_hh.append(HH)
        ll_list.append(np.stack(ch_ll, axis=0))
        lh_list.append(np.stack(ch_lh, axis=0))
        hl_list.append(np.stack(ch_hl, axis=0))
        hh_list.append(np.stack(ch_hh, axis=0))

    to_tensor = lambda lst: torch.from_numpy(np.stack(lst, axis=0)).float().to(device)
    return {
        "LL": to_tensor(ll_list),
        "LH": to_tensor(lh_list),
        "HL": to_tensor(hl_list),
        "HH": to_tensor(hh_list),
    }


def idwt2d(bands: dict, wavelet: str = "haar") -> torch.Tensor:
    """
    Reconstruct an image from DWT subbands via IDWT.

    Args:
        bands: dict with keys 'LL', 'LH', 'HL', 'HH', each (B, C, H, W)
        wavelet: must match the wavelet used in dwt2d

    Returns:
        (B, C, H*2, W*2) reconstructed tensor
    """
    device = bands["LL"].device
    B, C = bands["LL"].shape[:2]
    result = []
    for b in range(B):
        ch_out = []
        for c in range(C):
            LL = bands["LL"][b, c].cpu().numpy()
            LH = bands["LH"][b, c].cpu().numpy()
            HL = bands["HL"][b, c].cpu().numpy()
            HH = bands["HH"][b, c].cpu().numpy()
            rec = pywt.idwt2((LL, (LH, HL, HH)), wavelet)
            ch_out.append(rec)
        result.append(np.stack(ch_out, axis=0))
    return torch.from_numpy(np.stack(result, axis=0)).float().to(device)


def verify_reconstruction(x: torch.Tensor, wavelet: str = "haar", tol: float = 1e-5) -> float:
    """Sanity-check: measures max absolute error of DWT → IDWT round-trip."""
    bands = dwt2d(x, wavelet)
    x_rec = idwt2d(bands, wavelet)
    # Crop in case of odd-dimension padding
    h, w = x.shape[-2], x.shape[-1]
    err = (x - x_rec[..., :h, :w]).abs().max().item()
    return err

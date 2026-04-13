"""
hf_predictor.py — Lightweight high-frequency subband predictor for WG-DM.
Takes the restored LL band as input and predicts LH, HL, HH subbands.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
        )

    def forward(self, x):
        return x + self.net(x)


class HFPredictor(nn.Module):
    """
    Predicts the three high-frequency DWT subbands (LH, HL, HH) from the
    restored LL approximation subband. Architecture: lightweight encoder-decoder
    with skip connections, conditioned entirely on X_LL.

    Input:  (B, in_ch, H, W)   — restored LL subband
    Output: (B, in_ch*3, H, W) — concatenated LH, HL, HH predictions
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 48, n_res_blocks: int = 4):
        super().__init__()
        out_ch = in_ch * 3  # predict LH, HL, HH jointly

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.SiLU(),
        )

        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
                nn.SiLU(),
            ),
            nn.Sequential(
                nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
                nn.SiLU(),
            ),
        ])

        self.bottleneck = nn.Sequential(
            *[ResidualBlock(base_ch * 4) for _ in range(n_res_blocks)]
        )

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1),
                nn.SiLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(base_ch * 4, base_ch, 4, stride=2, padding=1),  # +skip
                nn.SiLU(),
            ),
        ])

        self.out = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),  # +stem skip
            nn.SiLU(),
            nn.Conv2d(base_ch, out_ch, 1),
        )

    def forward(self, ll: torch.Tensor) -> dict:
        s0 = self.stem(ll)                          # base_ch

        e1 = self.encoder[0](s0)                   # base_ch*2, H/2
        e2 = self.encoder[1](e1)                   # base_ch*4, H/4

        m = self.bottleneck(e2)                    # base_ch*4, H/4

        d1 = self.decoder[0](m)                    # base_ch*2, H/2
        d2 = self.decoder[1](torch.cat([d1, e1], dim=1))  # base_ch, H

        out_raw = self.out(torch.cat([d2, s0], dim=1))    # in_ch*3, H, W

        # Split into three subbands
        C = ll.shape[1]
        lh, hl, hh = out_raw[:, :C], out_raw[:, C:2*C], out_raw[:, 2*C:]
        return {"LH": lh, "HL": hl, "HH": hh}


class HFLoss(nn.Module):
    """L1 loss on each predicted HF subband."""
    def forward(self, pred: dict, target: dict) -> torch.Tensor:
        loss = 0.0
        for key in ("LH", "HL", "HH"):
            loss = loss + F.l1_loss(pred[key], target[key])
        return loss / 3.0

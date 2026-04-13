"""
diffusion.py — DDPM noise schedule + U-Net for WG-DM.
Diffusion operates only on the LL subband (half spatial resolution).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────
# Noise schedule (linear beta schedule)
# ─────────────────────────────────────────────

def make_beta_schedule(T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "alpha_bar_prev": alpha_bar_prev,
        "sqrt_alpha_bar": alpha_bar.sqrt(),
        "sqrt_one_minus_alpha_bar": (1.0 - alpha_bar).sqrt(),
        "posterior_variance": betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar),
    }


# ─────────────────────────────────────────────
# Sinusoidal timestep embedding
# ─────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        emb = t[:, None].float() * freqs[None]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ─────────────────────────────────────────────
# Basic ResBlock with time conditioning
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# ─────────────────────────────────────────────
# Lightweight U-Net denoiser (operates on LL band)
# ─────────────────────────────────────────────

class UNetDenoiser(nn.Module):
    """
    A small U-Net that predicts the noise ε from a noisy LL subband x_t.
    Input channels: in_ch (e.g. 3 for RGB LL band, optionally +3 for conditioning on degraded LL).
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Encoder
        self.enc1 = ResBlock(in_ch, base_ch, time_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, 2, 1)          # /2

        self.enc2 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, 2, 1)  # /4

        # Bottleneck
        self.mid1 = ResBlock(base_ch * 2, base_ch * 4, time_dim)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_dim)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1)
        self.dec2 = ResBlock(base_ch * 4, base_ch * 2, time_dim)    # cat skip

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1)
        self.dec1 = ResBlock(base_ch * 2, base_ch, time_dim)        # cat skip

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)

        m = self.mid2(self.mid1(self.down2(e2), t_emb), t_emb)

        d2 = self.dec2(torch.cat([self.up2(m), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)

        return self.out_conv(F.silu(self.out_norm(d1)))


# ─────────────────────────────────────────────
# DDPM forward / reverse step helpers
# ─────────────────────────────────────────────

def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: dict, noise: torch.Tensor = None) -> torch.Tensor:
    """Sample x_t from q(x_t | x_0) = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε"""
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab = schedule["sqrt_alpha_bar"][t].to(x0.device)[:, None, None, None]
    sqrt_1mab = schedule["sqrt_one_minus_alpha_bar"][t].to(x0.device)[:, None, None, None]
    return sqrt_ab * x0 + sqrt_1mab * noise, noise


def p_sample(model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, schedule: dict, cond: torch.Tensor = None) -> torch.Tensor:
    """One reverse diffusion step using the denoiser."""
    inp = torch.cat([x_t, cond], dim=1) if cond is not None else x_t
    eps_pred = model(inp, t)

    alpha = schedule["alphas"][t].to(x_t.device)[:, None, None, None]
    alpha_bar = schedule["alpha_bar"][t].to(x_t.device)[:, None, None, None]
    beta = schedule["betas"][t].to(x_t.device)[:, None, None, None]
    post_var = schedule["posterior_variance"][t].to(x_t.device)[:, None, None, None]

    mean = (x_t - beta / (1 - alpha_bar).sqrt() * eps_pred) / alpha.sqrt()
    z = torch.randn_like(x_t) if (t > 0).any() else torch.zeros_like(x_t)
    return mean + post_var.sqrt() * z


@torch.no_grad()
def ddpm_sample(model: nn.Module, shape: tuple, schedule: dict,
                cond: torch.Tensor = None, T: int = 1000, device: str = "cuda") -> torch.Tensor:
    """Full reverse diffusion loop (DDPM sampler)."""
    x = torch.randn(shape, device=device)
    for i in reversed(range(T)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        x = p_sample(model, x, t, schedule, cond=cond)
    return x

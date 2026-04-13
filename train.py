"""
train.py — Joint training loop for WG-DM.

Stage 1: Train UNetDenoiser on LL subbands (diffusion loss)
Stage 2: Train HFPredictor conditioned on restored LL (HF reconstruction loss)

Run: python train.py --config configs/train.yaml
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from wavelet import dwt2d, idwt2d
from diffusion import make_beta_schedule, q_sample, UNetDenoiser
from hf_predictor import HFPredictor, HFLoss
from metrics import compute_psnr, compute_ssim


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_dataloader(cfg: dict, split: str) -> DataLoader:
    tfm = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),   # → [-1, 1]
    ])
    dataset = ImageFolder(os.path.join(cfg["data_root"], split), transform=tfm)
    return DataLoader(dataset, batch_size=cfg["batch_size"],
                      shuffle=(split == "train"), num_workers=4, pin_memory=True)


# ─────────────────────────────────────────────
# Stage 1: Train diffusion model on LL band
# ─────────────────────────────────────────────

def train_diffusion(cfg: dict, device: torch.device):
    schedule = make_beta_schedule(T=cfg["T"])
    model = UNetDenoiser(
        in_ch=cfg["in_ch"] * 2,       # noisy LL + degraded LL conditioning
        base_ch=cfg["unet_base_ch"],
        time_dim=cfg["time_dim"],
    ).to(device)

    opt = AdamW(model.parameters(), lr=cfg["lr"])
    sched = CosineAnnealingLR(opt, T_max=cfg["epochs"])
    loader = get_dataloader(cfg, "train")

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    model.train()
    for epoch in range(cfg["epochs"]):
        total_loss = 0.0
        for batch, _ in tqdm(loader, desc=f"Diffusion epoch {epoch+1}"):
            hr = batch.to(device)

            # Simulate a degraded input (Gaussian blur + noise as SR proxy)
            lr_sim = F.avg_pool2d(hr, 2, 2)
            lr_sim = F.interpolate(lr_sim, size=hr.shape[-2:], mode="bilinear", align_corners=False)

            # Decompose GT high-res into DWT subbands
            bands_hr = dwt2d(hr)
            ll_hr = bands_hr["LL"]                 # (B, C, H/2, W/2)

            # Conditioning: LL of degraded image
            bands_lr = dwt2d(lr_sim)
            ll_cond = bands_lr["LL"]

            # Sample random timestep and add noise
            B = hr.shape[0]
            t = torch.randint(0, cfg["T"], (B,), device=device)
            x_t, noise = q_sample(ll_hr, t, schedule)

            # Predict noise (conditioned on degraded LL)
            inp = torch.cat([x_t, ll_cond], dim=1)
            eps_pred = model(inp, t)

            loss = F.mse_loss(eps_pred, noise)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        sched.step()
        avg = total_loss / len(loader)
        print(f"[Diffusion] Epoch {epoch+1}/{cfg['epochs']}  loss={avg:.4f}")

        if (epoch + 1) % cfg["save_every"] == 0:
            torch.save(model.state_dict(),
                       os.path.join(cfg["ckpt_dir"], f"diffusion_ep{epoch+1}.pt"))

    return model


# ─────────────────────────────────────────────
# Stage 2: Train HF predictor
# ─────────────────────────────────────────────

def train_hf_predictor(cfg: dict, device: torch.device, diff_model: UNetDenoiser):
    from diffusion import ddpm_sample

    hf_model = HFPredictor(
        in_ch=cfg["in_ch"],
        base_ch=cfg["hf_base_ch"],
        n_res_blocks=cfg["hf_res_blocks"],
    ).to(device)
    hf_loss_fn = HFLoss()

    opt = AdamW(hf_model.parameters(), lr=cfg["lr"])
    sched = CosineAnnealingLR(opt, T_max=cfg["hf_epochs"])
    loader = get_dataloader(cfg, "train")
    schedule = make_beta_schedule(T=cfg["T"])

    diff_model.eval()
    hf_model.train()
    for epoch in range(cfg["hf_epochs"]):
        total_loss = 0.0
        for batch, _ in tqdm(loader, desc=f"HF epoch {epoch+1}"):
            hr = batch.to(device)
            bands_hr = dwt2d(hr)

            # Generate restored LL via diffusion (quick sample with fewer steps)
            bands_lr = dwt2d(
                F.interpolate(
                    F.avg_pool2d(hr, 2, 2),
                    size=hr.shape[-2:], mode="bilinear", align_corners=False
                )
            )
            ll_cond = bands_lr["LL"]
            B, C, h, w = ll_cond.shape

            with torch.no_grad():
                ll_restored = ddpm_sample(
                    diff_model,
                    shape=(B, C, h, w),
                    schedule=schedule,
                    cond=ll_cond,
                    T=cfg["T"],
                    device=str(device),
                )

            hf_pred = hf_model(ll_restored)
            hf_target = {k: bands_hr[k] for k in ("LH", "HL", "HH")}
            loss = hf_loss_fn(hf_pred, hf_target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hf_model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        sched.step()
        avg = total_loss / len(loader)
        print(f"[HF Pred] Epoch {epoch+1}/{cfg['hf_epochs']}  loss={avg:.4f}")

    torch.save(hf_model.state_dict(), os.path.join(cfg["ckpt_dir"], "hf_predictor_final.pt"))
    return hf_model


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(cfg: dict, device: torch.device, diff_model, hf_model):
    from diffusion import ddpm_sample
    schedule = make_beta_schedule(T=cfg["T"])
    loader = get_dataloader(cfg, "val")
    diff_model.eval()
    hf_model.eval()

    psnr_vals, ssim_vals = [], []
    for batch, _ in tqdm(loader, desc="Evaluating"):
        hr = batch.to(device)
        lr_sim = F.avg_pool2d(hr, 2, 2)
        lr_up = F.interpolate(lr_sim, size=hr.shape[-2:], mode="bilinear", align_corners=False)
        bands_lr = dwt2d(lr_up)
        ll_cond = bands_lr["LL"]
        B, C, h, w = ll_cond.shape

        ll_restored = ddpm_sample(diff_model, (B, C, h, w), schedule,
                                   cond=ll_cond, T=cfg["T"], device=str(device))
        hf_pred = hf_model(ll_restored)
        bands_pred = {"LL": ll_restored, **hf_pred}
        x_hat = idwt2d(bands_pred)

        # Align shapes
        x_hat = x_hat[..., :hr.shape[-2], :hr.shape[-1]]
        x_hat_01 = (x_hat.clamp(-1, 1) + 1) / 2
        hr_01 = (hr + 1) / 2

        psnr_vals.append(compute_psnr(x_hat_01, hr_01))
        ssim_vals.append(compute_ssim(x_hat_01, hr_01))

    print(f"PSNR: {sum(psnr_vals)/len(psnr_vals):.2f} dB")
    print(f"SSIM: {sum(ssim_vals)/len(ssim_vals):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--stage", choices=["diffusion", "hf", "eval", "all"], default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.stage in ("diffusion", "all"):
        diff_model = train_diffusion(cfg, device)

    if args.stage in ("hf", "all"):
        if args.stage == "hf":
            diff_model = UNetDenoiser(cfg["in_ch"]*2, cfg["unet_base_ch"], cfg["time_dim"]).to(device)
            diff_model.load_state_dict(torch.load(cfg["diff_ckpt"]))
        hf_model = train_hf_predictor(cfg, device, diff_model)

    if args.stage in ("eval", "all"):
        if args.stage == "eval":
            diff_model = UNetDenoiser(cfg["in_ch"]*2, cfg["unet_base_ch"], cfg["time_dim"]).to(device)
            diff_model.load_state_dict(torch.load(cfg["diff_ckpt"]))
            hf_model = HFPredictor(cfg["in_ch"], cfg["hf_base_ch"], cfg["hf_res_blocks"]).to(device)
            hf_model.load_state_dict(torch.load(cfg["hf_ckpt"]))
        evaluate(cfg, device, diff_model, hf_model)

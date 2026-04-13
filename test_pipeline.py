"""
test_pipeline.py — Smoke test for the WG-DM pipeline without training.
Verifies DWT round-trip, model forward passes, and full inference pipeline.

Run: python test_pipeline.py
"""

import torch
from wavelet import dwt2d, idwt2d, verify_reconstruction
from diffusion import make_beta_schedule, q_sample, UNetDenoiser, ddpm_sample
from hf_predictor import HFPredictor

B, C, H, W = 2, 3, 256, 256
device = torch.device("cpu")


def test_wavelet():
    x = torch.randn(B, C, H, W)
    err = verify_reconstruction(x, wavelet="haar")
    assert err < 1e-5, f"DWT round-trip error too high: {err}"
    print(f"[PASS] DWT round-trip max error: {err:.2e}")

    bands = dwt2d(x)
    for key in ("LL", "LH", "HL", "HH"):
        assert bands[key].shape == (B, C, H//2, W//2), f"Wrong shape for {key}"
    print(f"[PASS] DWT subband shapes: {bands['LL'].shape}")


def test_diffusion():
    schedule = make_beta_schedule(T=10)   # small T for speed
    model = UNetDenoiser(in_ch=C*2, base_ch=16, time_dim=64).to(device)

    x = torch.randn(B, C, H//2, W//2)
    cond = torch.randn(B, C, H//2, W//2)
    t = torch.randint(0, 10, (B,))
    x_t, noise = q_sample(x, t, schedule)
    inp = torch.cat([x_t, cond], dim=1)
    eps = model(inp, t)
    assert eps.shape == x.shape, f"Denoiser output shape mismatch: {eps.shape}"
    print(f"[PASS] UNet denoiser output shape: {eps.shape}")


def test_hf_predictor():
    model = HFPredictor(in_ch=C, base_ch=16, n_res_blocks=2).to(device)
    ll = torch.randn(B, C, H//2, W//2)
    out = model(ll)
    for key in ("LH", "HL", "HH"):
        assert out[key].shape == (B, C, H//2, W//2), f"HF pred shape mismatch for {key}"
    print(f"[PASS] HF predictor output shapes: {out['LH'].shape}")


def test_full_pipeline():
    """End-to-end: degrade → DWT → diffuse LL → predict HF → IDWT → reconstruct."""
    import torch.nn.functional as F

    schedule = make_beta_schedule(T=5)
    diff_model = UNetDenoiser(in_ch=C*2, base_ch=16, time_dim=64).to(device)
    hf_model = HFPredictor(in_ch=C, base_ch=16, n_res_blocks=2).to(device)

    hr = torch.randn(B, C, H, W)
    lr = F.avg_pool2d(hr, 2, 2)
    lr_up = F.interpolate(lr, size=(H, W), mode="bilinear", align_corners=False)

    bands_lr = dwt2d(lr_up)
    ll_cond = bands_lr["LL"]

    ll_restored = ddpm_sample(diff_model, shape=(B, C, H//2, W//2),
                               schedule=schedule, cond=ll_cond, T=5, device="cpu")
    hf_pred = hf_model(ll_restored)
    bands_pred = {"LL": ll_restored, **hf_pred}
    x_hat = idwt2d(bands_pred)[..., :H, :W]

    assert x_hat.shape == hr.shape, f"Output shape mismatch: {x_hat.shape}"
    print(f"[PASS] Full pipeline output shape: {x_hat.shape}")


if __name__ == "__main__":
    print("=== WG-DM Pipeline Smoke Test ===")
    test_wavelet()
    # test_diffusion()
    # test_hf_predictor()
    test_full_pipeline()
    print("\nAll tests passed!")

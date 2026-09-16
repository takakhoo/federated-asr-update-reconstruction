#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

torch = None
np = None
plt = None
F = None


def _load_core_dependencies():
    global torch, np, plt, F
    try:
        import torch as _torch
        import numpy as _np
        import matplotlib.pyplot as _plt
        import torch.nn.functional as _F
    except ImportError as exc:
        raise RuntimeError(
            "Missing reconstruction dependencies. Install requirements.txt first."
        ) from exc
    torch, np, plt, F = _torch, _np, _plt, _F


def tv_2d(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (x[:, 1:, :] - x[:, :-1, :]).abs().mean()


def save_img(arr: np.ndarray, step: int, out_dir: str, title: str):
    os.makedirs(out_dir, exist_ok=True)
    vmin, vmax = np.percentile(arr, 1), np.percentile(arr, 99)
    disp = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)
    plt.figure(figsize=(10, 3))
    plt.imshow(disp, aspect="auto", origin="lower")
    plt.title(f"{title} step {step}")
    plt.tight_layout()
    out = os.path.join(out_dir, f"recon_step_{step:04d}.png")
    plt.savefig(out, dpi=150)
    plt.close()


def run_ls_mode(blob_path: str, out_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(blob_path, map_location="cpu")
    dW = saved["dW"].to(device)  # (V, D)
    dlogits = saved["dlogits"].to(device)  # (B, T, V) or similar

    V, D = dW.shape
    # Flatten time/batch into N
    Glog = dlogits.reshape(-1, dlogits.shape[-1])  # (N, V)
    # Optionally drop zero rows (padding)
    nonzero_mask = (Glog.abs().sum(dim=1) > 0)
    if nonzero_mask.any():
        Glog = Glog[nonzero_mask]
    # Compute per-frame feature estimates via least squares: H_hat = pinv(Glog^T) @ dW
    pinv = torch.linalg.pinv(Glog.t())  # (N, V)
    H_hat = pinv @ dW  # (N, D)
    H_hat_np = H_hat.detach().cpu().numpy().T  # (D, N) for display
    save_img(H_hat_np, step=0, out_dir=out_dir, title="H_hat (LS)")

    # Also compute average feature hbar
    sum_g = Glog.sum(dim=0)  # (V,)
    denom = float((sum_g @ sum_g).detach().cpu().numpy()) + 1e-12
    hbar = (sum_g.unsqueeze(0) @ dW).squeeze(0) / denom  # (D,)
    # Visualize hbar as a 1xD heatmap
    save_img(hbar.detach().cpu().numpy()[None, :], step=0, out_dir=out_dir, title="hbar (LS)")

    print(f"LS mode done. Saved H_hat and hbar visuals to {out_dir}")


def build_sb_modules(sb_yaml: str, device: torch.device):
    try:
        from hyperpyyaml import load_hyperpyyaml
    except ImportError as exc:
        raise RuntimeError(
            "hyperpyyaml is required for waveform mode; install requirements.txt."
        ) from exc
    with open(sb_yaml) as fin:
        params = load_hyperpyyaml(fin)
    modules = params["modules"]
    modules = modules.to(device)
    # Freeze wav2vec2 and enc
    for p in modules.wav2vec2.parameters():
        p.requires_grad_(False)
    for p in modules.enc.parameters():
        p.requires_grad_(False)
    return modules


def run_wave_mode(blob_path: str, sb_yaml: str, steps: int, lr: float, out_dir: str, wav_seconds: float, lambda_dlogits: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(blob_path, map_location="cpu")
    dW_star = saved["dW"].to(device)  # (V, D)
    W = saved["W"].to(device)
    b = saved["b"].to(device) if saved.get("b") is not None else None
    targets = saved["targets"].to(device)
    target_lens = saved["target_lens"].to(device)
    # dlogits target is optional
    dlogits_star = saved.get("dlogits")
    if dlogits_star is not None:
        dlogits_star = dlogits_star.to(device)

    V, D = dW_star.shape

    modules = build_sb_modules(sb_yaml, device)
    # Replace head with a torch.nn.Linear we control (same weights)
    head = torch.nn.Linear(D, V, bias=(b is not None)).to(device)
    with torch.no_grad():
        head.weight.copy_(W)
        if b is not None:
            head.bias.copy_(b)
    for p in head.parameters():
        p.requires_grad_(True)

    # Learnable waveform z
    sr = 16000
    num_samples = int(sr * wav_seconds)
    z = torch.nn.Parameter(0.01 * torch.randn(1, num_samples, device=device))
    opt = torch.optim.Adam([z], lr=lr)

    for step in range(steps):
        opt.zero_grad()
        # Forward
        feats = modules.wav2vec2(z)
        enc = modules.enc(feats)
        logits = head(enc)  # (B, T, V)
        p_ctc = F.log_softmax(logits, dim=-1)
        # For CTCLoss, use input lens = ones (no padding)
        input_lens = torch.ones(p_ctc.shape[0], device=device)

        # Compute CTC loss
        # SpeechBrain ctc_loss signature: (log_probs, targets, input_lens, target_lens)
        from speechbrain.nnet.losses import ctc_loss as sb_ctc_loss

        loss_ctc = sb_ctc_loss(p_ctc, targets, input_lens, target_lens)

        # Compute grads wrt head.weight and logits (create_graph to allow optimizing z)
        dW_cur = torch.autograd.grad(loss_ctc, head.weight, create_graph=True)[0]
        loss_g = torch.mean((dW_cur - dW_star) ** 2)

        loss_dlog = torch.tensor(0.0, device=device)
        if lambda_dlogits > 0.0 and dlogits_star is not None:
            dlogits_cur = torch.autograd.grad(loss_ctc, logits, create_graph=True)[0]
            # If time dims differ, compare frame-wise means
            if dlogits_cur.shape[1] != dlogits_star.shape[1]:
                loss_dlog = (dlogits_cur.mean(dim=1) - dlogits_star.mean(dim=1)).pow(2).mean()
            else:
                loss_dlog = (dlogits_cur - dlogits_star).pow(2).mean()
            loss_g = loss_g + lambda_dlogits * loss_dlog

        loss = loss_g
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == steps - 1:
            # Display a simple spectrogram-like map of z
            z_np = z.detach().cpu().numpy()
            save_img(z_np, step=step, out_dir=out_dir, title="waveform (raw)")

    print(f"Waveform mode done. Saved visuals to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grads", required=True, help="Path to saved grads blob (.pt)")
    ap.add_argument("--mode", choices=["ls", "waveform"], default="ls")
    ap.add_argument("--sb_config", default=None, help="SpeechBrain YAML; required for waveform mode")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--out_dir", default="viz")
    ap.add_argument("--wav_seconds", type=float, default=4.0)
    ap.add_argument("--lambda_dlogits", type=float, default=0.0)
    args = ap.parse_args()

    _load_core_dependencies()

    if args.mode == "ls":
        run_ls_mode(args.grads, args.out_dir)
    else:
        if not args.sb_config:
            ap.error("--sb_config is required when --mode waveform")
        run_wave_mode(args.grads, args.sb_config, args.steps, args.lr, args.out_dir, args.wav_seconds, args.lambda_dlogits)


if __name__ == "__main__":
    main()

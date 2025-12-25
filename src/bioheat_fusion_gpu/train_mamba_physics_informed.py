"""
Mamba + Physics-Informed (PINN-style) Training

Goal: keep the same working bioheat PINN-style setup (data+physics+smooth loss),
but replace the feature-UNet with a Mamba-based sequence model so we can ingest
many more frames and still get crisp outputs.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _get_cfg_section(cfg: DictConfig, key: str):
    if hasattr(cfg, "experiments") and hasattr(cfg.experiments, key):
        return getattr(cfg.experiments, key)
    return getattr(cfg, key)


def load_training_sample(args):
    """
    Load frames, mask and compute target bioheat for one measurement.

    Returns: (frames, target_bioheat, mask) all on CPU
      - frames: (T, H, W)
      - target_bioheat: (H, W)
      - mask: (H, W)
    """
    (pid, patient, h5_dir, mask_path, window_size, num_frames, target_hw, preprocess_device_str) = args

    import numpy as np
    import torch
    import torch.nn.functional as F

    from .bioheat_torch import compute_bioheat
    from .data_io import load_focus, load_frames_h5, load_mask
    from .fuse import get_mm_per_pixel

    device = torch.device(preprocess_device_str)

    try:
        focus_mm = load_focus(h5_dir, patient)
        frames_np = load_frames_h5(h5_dir, pid)
        if frames_np is None:
            return None, pid

        n_frames, h, w = frames_np.shape
        if n_frames < window_size * 2:
            return None, pid

        # Load mask at native resolution
        mask_np = load_mask(pid, (h, w), Path(mask_path))

        frames_t = torch.from_numpy(frames_np.astype(np.float32)).to(device)
        mask_t = torch.from_numpy(mask_np.astype(np.float32)).to(device)

        # Target: last window (cleanest bioheat)
        late_start = n_frames - window_size
        late_end = n_frames

        # Pixel size for correct Laplacian scaling (match target resolution)
        target_h, target_w = target_hw
        mm_per_px = get_mm_per_pixel(focus_mm, target_w, target_h)

        target_bioheat = compute_bioheat(
            frames_t,
            mask_t,
            late_start,
            late_end,
            smoothing_sigma=2.0,
            pixel_size_mm=mm_per_px,
            alpha=0.00014,
        )

        # Resize target + mask
        if target_bioheat.shape != (target_h, target_w):
            target_bioheat = F.interpolate(
                target_bioheat.unsqueeze(0).unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        if mask_t.shape != (target_h, target_w):
            mask_t = F.interpolate(
                mask_t.unsqueeze(0).unsqueeze(0),
                size=(target_h, target_w),
                mode="nearest",
            ).squeeze(0).squeeze(0)

        # Input: subsample/pad to requested num_frames and resize
        if n_frames > num_frames:
            indices = torch.linspace(0, n_frames - 1, num_frames).long()
            frames_t = frames_t[indices]
        elif n_frames < num_frames:
            pad = num_frames - n_frames
            frames_t = torch.cat([frames_t, frames_t[-1:].repeat(pad, 1, 1)], dim=0)

        if frames_t.shape[-2:] != (target_h, target_w):
            frames_t = F.interpolate(
                frames_t.unsqueeze(1),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        out = (frames_t.cpu(), target_bioheat.cpu(), mask_t.cpu())

        del frames_t, mask_t, target_bioheat
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return out, pid
    except Exception as e:
        log.warning(f"Error loading {pid}: {e}")
        return None, pid


def collect_training_data(cfg: DictConfig):
    import torch

    data_cfg = _get_cfg_section(cfg, "data")
    train_cfg = _get_cfg_section(cfg, "training")

    h5_dir = Path(cfg.paths.h5_dir)
    mask_path = Path(cfg.paths.mask_path)

    window_size = int(data_cfg.get("window_size", 100))
    num_frames = int(data_cfg.get("num_frames", 200))
    target_h = int(data_cfg.get("target_h", 240))
    target_w = int(data_cfg.get("target_w", 320))
    target_hw = (target_h, target_w)

    num_workers = int(data_cfg.get("num_workers", 8))
    preprocess_device = str(data_cfg.get("preprocess_device", "cpu"))
    max_tasks = data_cfg.get("max_tasks", None)
    max_samples = data_cfg.get("max_samples", None)
    max_tasks = int(max_tasks) if max_tasks is not None else None
    max_samples = int(max_samples) if max_samples is not None else None
    skip_pids = list(data_cfg.get("skip_pids", []))
    device_str = str(train_cfg.get("device", cfg.training.device))

    patients = sorted([p.stem for p in h5_dir.glob("P*.h5")])
    measurements = ["M01", "M02", "M03", "M04"]

    tasks = []
    for patient in patients:
        for meas in measurements:
            pid = f"{patient}{meas}"
            if pid in skip_pids:
                continue
            tasks.append((pid, patient, h5_dir, mask_path, window_size, num_frames, target_hw, preprocess_device))
            if max_tasks is not None and len(tasks) >= max_tasks:
                break
        if max_tasks is not None and len(tasks) >= max_tasks:
            break

    log.info(f"Found {len(patients)} patients in {h5_dir}")
    log.info(f"Processing {len(tasks)} measurement(s) with {num_workers} worker(s)")
    log.info(f"Target size: {target_h}x{target_w}, num_frames={num_frames}, window_size={window_size}")
    log.info(f"Preprocessing device: {preprocess_device}")
    if max_samples is not None:
        log.info(f"Stopping after {max_samples} collected sample(s)")

    train_data = []
    completed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_training_sample, task): task[0] for task in tasks}
        for future in as_completed(futures):
            pid = futures[future]
            completed += 1
            try:
                result, _ = future.result()
            except Exception as e:
                log.error(f"[{completed}/{len(tasks)}] {pid}: error: {e}")
                continue

            if result is None:
                log.info(f"[{completed}/{len(tasks)}] {pid}: no data")
                continue

            frames, target, mask = result
            train_data.append((frames, target, mask))
            log.info(f"[{completed}/{len(tasks)}] {pid}: frames={tuple(frames.shape)} target={tuple(target.shape)}")

            if max_samples is not None and len(train_data) >= max_samples:
                for f in futures:
                    f.cancel()
                break

    log.info(f"Collected {len(train_data)} training samples")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return train_data


def train_mamba_physics_informed(
    model,
    train_data: list,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    save_path: Path,
    early_stop_patience: int,
    data_weight: float,
    physics_weight: float,
    smooth_weight: float,
    edge_weight: float,
    alpha: float,
    pixel_size_mm: float,
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm

    from .physics_informed_net import PhysicsInformedLoss, compute_temporal_derivative

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        log.info(f"Using DataParallel with {n_gpus} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device_t)

    criterion = PhysicsInformedLoss(
        data_weight=data_weight,
        physics_weight=physics_weight,
        smooth_weight=smooth_weight,
        edge_weight=edge_weight,
        alpha=alpha,
        pixel_size_mm=pixel_size_mm,
    ).to(device_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    loss_history = []
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_losses = {"data": 0.0, "physics": 0.0, "smooth": 0.0, "edge": 0.0}
        n_samples = 0

        indices = np.random.permutation(len(train_data))
        pbar = tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for i in pbar:
            batch_idx = indices[i : i + batch_size]

            frames_list, target_list, mask_list = [], [], []
            for idx in batch_idx:
                frames, target, mask = train_data[idx]
                frames_list.append(frames)
                target_list.append(target)
                mask_list.append(mask)

            frames_batch = torch.stack(frames_list).to(device_t)  # (B, T, H, W)
            target_batch = torch.stack(target_list).to(device_t).unsqueeze(1)  # (B, 1, H, W)
            mask_batch = torch.stack(mask_list).to(device_t).unsqueeze(1)  # (B, 1, H, W)

            # Physics features from frames
            mean_temp = frames_batch.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            observed_slope = compute_temporal_derivative(frames_batch)  # (B, 1, H, W)
            observed_slope = F.relu(observed_slope)

            # Normalize (match the working physics-informed setup)
            mean_norm = (mean_temp - mean_temp.mean()) / (mean_temp.std() + 1e-8)
            slope_norm = (observed_slope - observed_slope.mean()) / (observed_slope.std() + 1e-8)
            target_norm = (target_batch - target_batch.mean()) / (target_batch.std() + 1e-8)

            optimizer.zero_grad()

            pred = model(frames_batch)
            if pred.shape[-2:] != target_batch.shape[-2:]:
                pred = F.interpolate(pred, size=target_batch.shape[-2:], mode="bilinear", align_corners=True)

            pred_norm = (pred - pred.mean()) / (pred.std() + 1e-8)

            loss, parts = criterion(pred_norm, target_norm, slope_norm, mean_norm, mask_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            B = frames_batch.shape[0]
            epoch_loss += loss.item() * B
            for k in epoch_losses:
                if k in parts:
                    epoch_losses[k] += float(parts[k]) * B
            n_samples += B
            pbar.set_postfix(loss=loss.item())

        epoch_loss /= max(1, n_samples)
        for k in epoch_losses:
            epoch_losses[k] /= max(1, n_samples)
        loss_history.append(epoch_loss)
        scheduler.step()

        log.info(
            f"Epoch {epoch+1}: loss={epoch_loss:.6f} "
            f"(data={epoch_losses['data']:.4f}, physics={epoch_losses['physics']:.4f}, "
            f"smooth={epoch_losses['smooth']:.4f}, edge={epoch_losses['edge']:.4f})"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model_to_save.state_dict(),
                    "epoch": epoch,
                    "loss": best_loss,
                    "loss_components": epoch_losses,
                    "n_gpus": n_gpus,
                },
                save_path,
            )
            log.info("  Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                log.info(f"Early stopping at epoch {epoch+1}")
                break

    if isinstance(model, nn.DataParallel):
        model = model.module
    return model, loss_history


def run_training(cfg: DictConfig) -> None:
    import torch
    import matplotlib.pyplot as plt

    from .mamba_pinn_net import MambaPINNNet

    train_cfg = _get_cfg_section(cfg, "training")
    model_cfg = _get_cfg_section(cfg, "model")
    data_cfg = _get_cfg_section(cfg, "data")
    physics_cfg = cfg.get("physics", {})

    log.info("=" * 60)
    log.info("Mamba + Physics-Informed (PINN-style) Training")
    log.info("=" * 60)
    log.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    device_str = train_cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device_str = "cpu"

    log.info(f"Detected {torch.cuda.device_count()} GPU(s)")

    train_data = collect_training_data(cfg)
    if not train_data:
        log.error("No training data collected!")
        return

    base_channels = int(model_cfg.get("base_channels", 32))
    d_state = int(model_cfg.get("d_state", 16))
    n_mamba_layers = int(model_cfg.get("n_mamba_layers", 4))
    model = MambaPINNNet(base_channels=base_channels, d_state=d_state, n_mamba_layers=n_mamba_layers)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.paths.output_dir) / f"models_{timestamp}"
    save_path = out_dir / f"mamba_physics_informed_{timestamp}.pth"

    alpha = float(physics_cfg.get("alpha", 0.00014))

    # Pixel-size: keep the original default (0.5mm) unless overridden.
    # This matches the existing working physics-informed setup.
    pixel_size_mm = float(train_cfg.get("pixel_size_mm", 0.5))

    model, loss_history = train_mamba_physics_informed(
        model=model,
        train_data=train_data,
        epochs=int(train_cfg.get("epochs", 100)),
        batch_size=int(train_cfg.get("batch_size", 1)),
        lr=float(train_cfg.get("lr", 1e-4)),
        device=device_str,
        save_path=save_path,
        early_stop_patience=int(train_cfg.get("early_stop_patience", 25)),
        data_weight=float(train_cfg.get("data_weight", 1.0)),
        physics_weight=float(train_cfg.get("physics_weight", 0.5)),
        smooth_weight=float(train_cfg.get("smooth_weight", 0.1)),
        edge_weight=float(train_cfg.get("edge_weight", 0.0)),
        alpha=alpha,
        pixel_size_mm=pixel_size_mm,
    )

    plt.figure(figsize=(10, 4))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss (Data + Physics + Smooth + Edge)")
    plt.title("Mamba + Physics-Informed Training")
    plt.grid(True, alpha=0.3)
    plot_path = save_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    log.info(f"Training complete! Saved to {save_path}")
    log.info(f"Loss plot saved to {plot_path}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Hydra config loaded successfully")
    log.info(f"Working directory: {os.getcwd()}")
    run_training(cfg)


if __name__ == "__main__":
    main()

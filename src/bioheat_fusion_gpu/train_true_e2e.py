"""
TRUE End-to-End 3D Training (single-process)

This is the "known working" training path: it can use:
- single GPU (default), or
- torch.nn.DataParallel when multiple GPUs are visible.

It does NOT use DistributedDataParallel (DDP).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def load_training_sample(args):
    """Load frames and compute target bioheat for one measurement."""
    (pid, patient, h5_dir, mask_path, window_size, n_frames, device_str) = args

    import numpy as np
    import torch
    import torch.nn.functional as F

    from .bioheat_torch import compute_bioheat
    from .data_io import load_focus, load_frames_h5, load_mask
    from .fuse import get_mm_per_pixel

    device = torch.device(device_str)

    try:
        focus_mm = load_focus(h5_dir, patient)
        frames_np = load_frames_h5(h5_dir, pid)
        if frames_np is None:
            return None, pid

        total_frames, h, w = frames_np.shape
        if total_frames < window_size * 2:
            return None, pid

        mm_per_px = get_mm_per_pixel(focus_mm, w, h)
        mask_np = load_mask(pid, (h, w), mask_path)

        frames_t = torch.from_numpy(frames_np.astype(np.float32)).to(device)
        mask_t = torch.from_numpy(mask_np.astype(np.float32)).to(device)

        # Target from last window (cleanest bioheat)
        late_start = total_frames - window_size
        late_end = total_frames
        target_bioheat = compute_bioheat(
            frames_t,
            mask_t,
            late_start,
            late_end,
            smoothing_sigma=2.0,
            pixel_size_mm=mm_per_px,
            alpha=0.00014,
        )

        # Resize to standard size for 3D conv memory
        target_size = (240, 320)
        if target_bioheat.shape != target_size:
            target_bioheat = F.interpolate(
                target_bioheat.unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        # Subsample frames for 3D input
        step = max(1, total_frames // n_frames)
        frame_indices = list(range(0, total_frames, step))[:n_frames]
        input_frames = frames_t[frame_indices].cpu()  # (T, H, W)

        if input_frames.shape[-2:] != target_size:
            input_frames = F.interpolate(
                input_frames.unsqueeze(1),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        del frames_t, mask_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return (input_frames, target_bioheat.cpu()), pid
    except Exception as e:
        log.warning("Error loading %s: %s", pid, e)
        return None, pid


def collect_training_data(cfg: DictConfig, num_workers: int = 8):
    """Collect training data."""
    h5_dir = Path(cfg.paths.h5_dir)
    mask_path = Path(cfg.paths.mask_path)

    window_size = int(cfg.data.window_size)
    n_frames = int(cfg.data.get("n_input_frames", 50))
    skip_pids = list(cfg.data.skip_pids) if cfg.data.skip_pids else []

    device_str = str(cfg.training.device)
    if device_str == "cuda":
        device_str = "cuda"  # allow torch to resolve default device

    patients = sorted([p.stem for p in h5_dir.glob("P*.h5")])
    measurements = ["M01", "M02", "M03", "M04"]

    tasks = []
    for patient in patients:
        for meas in measurements:
            pid = f"{patient}{meas}"
            if pid in skip_pids:
                continue
            tasks.append((pid, patient, h5_dir, mask_path, window_size, n_frames, device_str))

    train_data = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_training_sample, task): task[0] for task in tasks}
        for future in as_completed(futures):
            pid = futures[future]
            try:
                result, _ = future.result()
                if result is not None:
                    train_data.append(result)
                else:
                    log.info("[skip] %s", pid)
            except Exception as e:
                log.error("Error processing %s: %s", pid, e)

    log.info("Collected %d training samples", len(train_data))
    return train_data


def run_training(cfg: DictConfig) -> None:
    import torch
    import torch.nn as nn

    from .true_e2e_3d import TrueEndToEnd3D, train_true_e2e

    log.info("=" * 60)
    log.info("TRUE End-to-End 3D Training (single-process)")
    log.info("=" * 60)
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    n_gpus = torch.cuda.device_count()
    log.info("Detected %d GPU(s)", n_gpus)

    num_workers = int(cfg.data.get("num_workers", 8))
    train_data = collect_training_data(cfg, num_workers=num_workers)
    if not train_data:
        log.error("No training data collected!")
        return

    model_path = Path(cfg.paths.output_dir) / "models" / f"true_e2e_3d_{max(1, n_gpus)}gpu.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    base_channels = int(cfg.model.get("base_channels", 16))
    model = TrueEndToEnd3D(base_channels=base_channels)

    if n_gpus > 1:
        model = nn.DataParallel(model)

    use_physics_loss = bool(cfg.training.get("use_physics_loss", True))

    model, loss_history = train_true_e2e(
        model=model,
        train_data=train_data,
        epochs=int(cfg.training.epochs),
        batch_size=int(cfg.training.get("batch_size", 2)),
        lr=float(cfg.training.lr),
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_path=model_path,
        early_stop_patience=int(cfg.training.early_stop_patience),
        use_physics_loss=use_physics_loss,
    )

    # Save a loss plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"True E2E 3D Training ({max(1, n_gpus)} GPU)")
        plt.grid(True, alpha=0.3)
        plt.savefig(model_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        log.info("Skipping loss plot: %s", e)

    log.info("Training complete! Model saved to %s", model_path)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("Hydra config loaded successfully")
    log.info("Working directory: %s", os.getcwd())
    run_training(cfg)


if __name__ == "__main__":
    main()


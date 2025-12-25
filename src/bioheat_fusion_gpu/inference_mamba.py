"""
Inference script for the Mamba-based models (thermal + artifact-aware).

Mimics the preprocessing used during training:
- 200 frames subsampled/padded per measurement
- resize to 240x320 prior to inference
- cooled-area mask from COCO annotations

Generates the same diagnostic plots as the TRUE E2E pipeline so we can
compare outputs apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Model + data helpers
# -----------------------------------------------------------------------------

def _strip_module_prefix(state: dict) -> dict:
    if not state:
        return state
    if any(k.startswith("module.") for k in state.keys()):
        return {k.removeprefix("module."): v for k, v in state.items()}
    return state


def _infer_mamba_pinn_hparams(state: dict) -> Tuple[int, int, int]:
    """
    Infer (base_channels, d_state, n_mamba_layers) from a MambaPINNNet state dict.
    """
    base_channels = 32
    d_state = 16
    n_mamba_layers = 4

    for k, v in state.items():
        if k == "spatial_encoder.encoder.0.weight" and hasattr(v, "shape") and len(v.shape) >= 1:
            base_channels = int(v.shape[0])
            break

    for k, v in state.items():
        if k.endswith("A_log") and k.startswith("mamba_layers.0.") and hasattr(v, "shape") and len(v.shape) == 1:
            d_state = int(v.shape[0])
            break

    layer_ids = set()
    for k in state.keys():
        if k.startswith("mamba_layers."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_ids.add(int(parts[1]))
    if layer_ids:
        n_mamba_layers = max(layer_ids) + 1

    return base_channels, d_state, n_mamba_layers


def load_mamba_model(model_path: str, model_kind: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load either the plain Mamba thermal net or the Artifact-aware variant.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if model_kind == "pinn":
        from .mamba_pinn_net import MambaPINNNet

        state = _strip_module_prefix(checkpoint["model_state_dict"])
        base_channels, d_state, n_mamba_layers = _infer_mamba_pinn_hparams(state)

        model = MambaPINNNet(
            base_channels=base_channels,
            d_state=d_state,
            n_mamba_layers=n_mamba_layers,
        )
    else:
        raise ValueError(f"Unsupported model_kind '{model_kind}'. Use 'pinn'.")

    state = _strip_module_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    print(f"Loaded {model_kind} Mamba model from {model_path}")
    epoch = checkpoint.get("epoch")
    loss = checkpoint.get("loss")
    if epoch is not None:
        print(f"  Epoch: {epoch}")
    if isinstance(loss, (float, int)):
        print(f"  Loss: {loss:.6f}")

    return model


def _read_first_available(dataset: h5py.File, candidate_paths: List[str]) -> Optional[np.ndarray]:
    for path in candidate_paths:
        if path in dataset:
            return dataset[path][:]
    return None


def load_frames_from_h5(h5_path: str, measurement: str) -> Optional[np.ndarray]:
    candidate_paths = [
        f"Measurements/Cooling/{measurement}/frames",
        f"Measurements/{measurement}/frames",
        f"measurements/{measurement}/frames",
        f"{measurement}/frames",
    ]
    try:
        with h5py.File(h5_path, "r") as f:
            frames = _read_first_available(f, candidate_paths)
            if frames is not None:
                print(f"    Frames loaded: {frames.shape}")
            return frames
    except Exception as exc:
        print(f"    Error reading frames from {h5_path}: {exc}")
        return None


def load_target_from_h5(h5_path: str, measurement: str) -> Optional[np.ndarray]:
    candidate_paths = [
        f"Measurements/Cooling/{measurement}/bioheat",
        f"Measurements/{measurement}/bioheat",
        f"measurements/{measurement}/bioheat",
        f"{measurement}/bioheat",
    ]
    try:
        with h5py.File(h5_path, "r") as f:
            target = _read_first_available(f, candidate_paths)
            if target is not None:
                print(f"    Target bioheat found: {target.shape}")
            return target
    except Exception as exc:
        print(f"    Error reading target from {h5_path}: {exc}")
        return None


def compute_late_frame_target(frames: np.ndarray, late_start_frac: float = 0.7) -> np.ndarray:
    """
    Matches the training target computation (late regression with ReLU).
    """
    n_frames = frames.shape[0]
    late_start = int(n_frames * late_start_frac)
    late_frames = frames[late_start:]

    t = np.arange(len(late_frames), dtype=np.float32)
    t = t - t.mean()
    t = t.reshape(-1, 1, 1)

    late_centered = late_frames - late_frames.mean(axis=0, keepdims=True)
    numerator = (t * late_centered).sum(axis=0)
    denominator = (t**2).sum() + 1e-8
    target = np.maximum(numerator / denominator, 0.0)
    return target


def load_mask_from_coco(mask_path: Optional[Path], patient_id: str, measurement: str, shape: Tuple[int, int]) -> np.ndarray:
    if mask_path is None or not mask_path.exists():
        return np.ones(shape, dtype=bool)

    try:
        with open(mask_path, "r") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"    Failed to read mask JSON: {exc}")
        return np.ones(shape, dtype=bool)

    mask = np.zeros(shape, dtype=np.uint8)
    pid_candidates = [f"{patient_id}{measurement}", f"{patient_id}_{measurement}"]

    image_id = None
    for img in data.get("images", []):
        name = img.get("file_name", "").lower()
        if any(pid.lower() in name for pid in pid_candidates):
            image_id = img.get("id")
            break

    if image_id is None:
        print(f"    Mask not found for {patient_id}/{measurement}; defaulting to full image.")
        return np.ones(shape, dtype=bool)

    for ann in data.get("annotations", []):
        if ann.get("image_id") == image_id and ann.get("segmentation"):
            for seg in ann["segmentation"]:
                pts = np.array(seg, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [pts], 1)

    if mask.sum() == 0:
        print(f"    Empty mask for {patient_id}/{measurement}; defaulting to full image.")
        return np.ones(shape, dtype=bool)

    return mask.astype(bool)


def prepare_frames_for_mamba(
    frames: np.ndarray,
    num_frames: int = 200,
    target_size: Tuple[int, int] = (240, 320),
) -> torch.Tensor:
    """
    Subsample/pad to num_frames and resize to target_size.

    Returns tensor with shape (1, num_frames, H, W).
    """
    frames_t = torch.from_numpy(frames.astype(np.float32))
    n_total = frames_t.shape[0]

    if n_total > num_frames:
        idx = torch.linspace(0, n_total - 1, num_frames).long()
        frames_t = frames_t[idx]
    elif n_total < num_frames:
        pad = num_frames - n_total
        frames_t = torch.cat([frames_t, frames_t[-1:].repeat(pad, 1, 1)], dim=0)

    if frames_t.shape[1:] != target_size:
        frames_t = F.interpolate(
            frames_t.unsqueeze(1), size=target_size, mode="bilinear", align_corners=True
        ).squeeze(1)

    return frames_t.unsqueeze(0)


def resize_to_shape(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=shape, mode="bilinear", align_corners=True)
    return tensor.squeeze().numpy()


def run_mamba_inference(
    model: torch.nn.Module,
    frames: np.ndarray,
    device: str,
    num_frames: int,
    target_size: Tuple[int, int],
) -> np.ndarray:
    model = model.to(device)
    x = prepare_frames_for_mamba(frames, num_frames=num_frames, target_size=target_size).to(device)

    with torch.no_grad():
        pred = model(x)  # Both models output (B, 1, H, W)

    return pred.squeeze().cpu().numpy()


# -----------------------------------------------------------------------------
# Visualization (reuse layout)
# -----------------------------------------------------------------------------

def normalize_with_mask(arr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    masked_vals = arr[mask]
    if masked_vals.size == 0:
        return np.zeros_like(arr), (0.0, 0.0)
    min_val = float(masked_vals.min())
    max_val = float(masked_vals.max())
    scale = max(max_val - min_val, 1e-8)
    norm = (arr - min_val) / scale
    norm = np.clip(norm, 0.0, 1.0)
    norm[~mask] = np.nan
    return norm, (min_val, max_val)


def visualize_mamba(
    frames: np.ndarray,
    target: np.ndarray,
    perfusion: np.ndarray,
    mask: np.ndarray,
    patient_id: str,
    measurement: str,
    output_dir: Path,
    model_label: str,
    clip_percentile: float = 99.0,
) -> Path:
    target_masked = np.where(mask, target, np.nan)
    perf_masked = np.where(mask, perfusion, np.nan)

    vmax_target = np.nanpercentile(target_masked, clip_percentile)
    vmax_perf = np.nanpercentile(perf_masked, clip_percentile)

    target_norm, (t_min, t_max) = normalize_with_mask(target, mask)
    perf_norm, (p_min, p_max) = normalize_with_mask(perfusion, mask)

    diff_norm = perf_norm - target_norm
    diff_norm[~mask] = np.nan
    vmax_diff = np.nanmax(np.abs(diff_norm))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ax = axes[0, 0]
    im = ax.imshow(target_masked, cmap="hot", vmin=0, vmax=vmax_target)
    ax.set_title(f"Target Bioheat\n0-{vmax_target:.3f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    im = ax.imshow(perf_masked, cmap="hot", vmin=0, vmax=vmax_perf)
    ax.set_title(f"{model_label} (clipped {clip_percentile:.1f}%)\n0-{vmax_perf:.3f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 2]
    mean_frame = frames.mean(axis=0)
    base = ax.imshow(mean_frame, cmap="hot")
    ax.imshow(mask, cmap="gray", alpha=0.35)
    ax.set_title("Mask overlay on mean frame")
    ax.axis("off")
    plt.colorbar(base, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    im = ax.imshow(target_norm, cmap="hot", vmin=0, vmax=1)
    ax.set_title(f"Target normalized\nmin={t_min:.3f}, max={t_max:.3f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    im = ax.imshow(perf_norm, cmap="hot", vmin=0, vmax=1)
    ax.set_title(f"{model_label} normalized\nmin={p_min:.3f}, max={p_max:.3f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 2]
    im = ax.imshow(diff_norm, cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff)
    ax.set_title("Normalized difference\n(Mamba - Target)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f"{patient_id}/{measurement} – {model_label}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{model_label.replace(' ', '_').lower()}_{patient_id}_{measurement}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"    Saved visualization: {save_path}")
    return save_path


# -----------------------------------------------------------------------------
# Inference loop
# -----------------------------------------------------------------------------

def run_inference_on_patient(
    model: torch.nn.Module,
    h5_path: Path,
    measurements: List[str],
    mask_path: Optional[Path],
    output_dir: Path,
    device: str,
    num_frames: int,
    target_size: Tuple[int, int],
    clip_percentile: float,
    model_label: str,
):
    patient_id = h5_path.stem
    print(f"\nProcessing {patient_id}...")

    for measurement in measurements:
        print(f"  Measurement {measurement}")
        frames = load_frames_from_h5(str(h5_path), measurement)
        if frames is None:
            print("    Skipping (frames unavailable).")
            continue

        original_shape = frames.shape[1:]
        target = load_target_from_h5(str(h5_path), measurement)
        if target is None:
            target = compute_late_frame_target(frames)
            print("    Target missing – computed from late frames.")

        if target.shape != original_shape:
            target = resize_to_shape(target, original_shape)

        perf = run_mamba_inference(
            model,
            frames,
            device=device,
            num_frames=num_frames,
            target_size=target_size,
        )
        if perf.shape != original_shape:
            perf = resize_to_shape(perf, original_shape)

        mask = load_mask_from_coco(mask_path, patient_id, measurement, original_shape)

        save_path = visualize_mamba(
            frames=frames,
            target=target,
            perfusion=perf,
            mask=mask,
            patient_id=patient_id,
            measurement=measurement,
            output_dir=output_dir,
            model_label=model_label,
            clip_percentile=clip_percentile,
        )

        masked_pred = perf[mask]
        masked_target = target[mask]
        print(f"    {model_label} stats (masked): min={masked_pred.min():.4f}, "
              f"max={masked_pred.max():.4f}, mean={masked_pred.mean():.4f}")
        print(f"    Target stats (masked):       min={masked_target.min():.4f}, "
              f"max={masked_target.max():.4f}, mean={masked_target.mean():.4f}")
        print(f"    Figure saved at {save_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mamba inference with masking + diagnostics.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to Mamba checkpoint (.pth).",
    )
    parser.add_argument(
        "--model_kind",
        type=str,
        choices=["pinn"],
        default="pinn",
        help="Which architecture to instantiate.",
    )
    parser.add_argument(
        "--h5_dir",
        type=str,
        default="/dodrio/scratch/projects/starting_2025_090/H5",
        help="Directory with patient H5 files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/dodrio/scratch/projects/starting_2025_090/output_n2n/mamba_inference",
        help="Directory for rendered figures.",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="/dodrio/scratch/projects/starting_2025_090/Annotations/coco_annotations_cooled_area.json",
        help="COCO mask annotations.",
    )
    parser.add_argument(
        "--patients",
        type=str,
        nargs="+",
        default=["P18", "P19", "P20"],
        help="Patient IDs to evaluate.",
    )
    parser.add_argument(
        "--measurements",
        type=str,
        nargs="+",
        default=["M01", "M02", "M03", "M04"],
        help="Measurement IDs.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=200,
        help="Number of frames fed to the Mamba model.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=(240, 320),
        help="Spatial size expected by the model (H W).",
    )
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=99.0,
        help="Percentile for clipping visualizations.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device.",
    )

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available – falling back to CPU.")
        device = "cpu"

    output_dir = Path(args.output_dir)
    mask_path = Path(args.mask_path) if args.mask_path else None
    h5_dir = Path(args.h5_dir)

    model = load_mamba_model(args.model, args.model_kind, device=device)
    if args.model_kind == "thermal":
        model_label = "Mamba Thermal"
    else:
        model_label = "Mamba PINN"

    for patient_id in args.patients:
        h5_path = h5_dir / f"{patient_id}.h5"
        if not h5_path.exists():
            print(f"\nSkipping {patient_id}: {h5_path} missing.")
            continue
        run_inference_on_patient(
            model=model,
            h5_path=h5_path,
            measurements=args.measurements,
            mask_path=mask_path,
            output_dir=output_dir,
            device=device,
            num_frames=args.num_frames,
            target_size=tuple(args.target_size),
            clip_percentile=args.clip_percentile,
            model_label=model_label,
        )

    print("\nMamba inference complete. Figures stored in:", output_dir)


if __name__ == "__main__":
    main()

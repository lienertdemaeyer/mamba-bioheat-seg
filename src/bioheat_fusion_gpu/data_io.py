import json
from pathlib import Path
import h5py
import numpy as np
import cv2


def load_frames_h5(h5_dir: Path, pid: str):
    base_id = pid[:3]
    meas_id = pid[3:]
    h5_path = h5_dir / f"{base_id}.h5"
    if not h5_path.exists():
        return None
    try:
        with h5py.File(h5_path, "r") as f:
            for path in [f"Measurements/Cooling/{meas_id}/frames", f"Cooling/{meas_id}/frames", f"Measurements/{meas_id}/frames"]:
                try:
                    obj = f
                    for part in path.split("/"):
                        obj = obj[part]
                    return obj[:].astype(np.float32)
                except Exception:
                    continue
    except Exception:
        return None
    return None


def load_focus(h5_dir: Path, base_id: str):
    h5_path = h5_dir / f"{base_id}.h5"
    if not h5_path.exists():
        return None
    try:
        with h5py.File(h5_path, "r") as f:
            if "metadata" in f and "focus_mm" in f["metadata"]:
                return float(f["metadata"]["focus_mm"][()])
            if "focus_mm" in f.attrs:
                return float(f.attrs["focus_mm"])
    except Exception:
        return None
    return None


def load_mask(pid: str, shape, mask_path: Path, apply_exclusions: bool = False, include_partial: bool = False):
    """Load mask from COCO JSON.
    
    Args:
        pid: Patient+measurement ID (e.g., "P01M01")
        shape: (height, width) tuple
        mask_path: Path to COCO JSON file
        apply_exclusions: If True, apply umbilicus/artefact exclusions (category_id=2,3).
                         If False (default), only use cooled_area as mask.
        include_partial: If True, also include partial_cooled_area (category_id=4) in the mask.
    """
    if not mask_path.exists():
        return np.ones(shape, dtype=bool)
    try:
        with open(mask_path, "r") as f:
            data = json.load(f)
        for img in data.get("images", []):
            if pid in img.get("file_name", ""):
                img_mask = np.zeros(shape, dtype=np.uint8)
                for ann in data.get("annotations", []):
                    if ann.get("image_id") == img.get("id"):
                        cat_id = ann.get("category_id")
                        if "segmentation" in ann and ann["segmentation"]:
                            for seg in ann["segmentation"]:
                                pts = np.array(seg, dtype=np.int32).reshape((-1, 2))
                                # category 1 = cooled_area (foreground)
                                if cat_id == 1:
                                    cv2.fillPoly(img_mask, [pts], 1)
                                # category 4 = partial_cooled_area (optional foreground)
                                elif include_partial and cat_id == 4:
                                    cv2.fillPoly(img_mask, [pts], 1)
                                # category 2 = umbilicus, 3 = artefact (exclusions)
                                elif apply_exclusions and cat_id in (2, 3):
                                    cv2.fillPoly(img_mask, [pts], 0)
                if img_mask.sum() > 0:
                    return img_mask.astype(bool)
    except Exception:
        return np.ones(shape, dtype=bool)
    return np.ones(shape, dtype=bool)

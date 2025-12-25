# Model Weights

This repository does not commit `*.pth` model files to Git to keep clones lightweight.
Instead, weights are published as **GitHub Release assets** and downloaded on demand.

## Recommended workflow

1. Create a GitHub release (e.g. tag `v0.1.0`).
2. Upload the model files as release assets (e.g. `mamba_physics_informed_20251223_114254.pth`).
3. Record their SHA256 checksums in `weights/manifest.json`.
4. Users download via `scripts/fetch_weights.py`.

## Download

```bash
python scripts/fetch_weights.py --repo OWNER/REPO --tag v0.1.0
```

This downloads all entries listed in `weights/manifest.json` into `weights/` and verifies SHA256 when provided.

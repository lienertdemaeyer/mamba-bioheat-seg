#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model weights from GitHub Releases.")
    parser.add_argument("--repo", required=True, help="GitHub repo in the form OWNER/REPO.")
    parser.add_argument("--tag", required=True, help="Release tag, e.g. v0.1.0.")
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parents[1] / "weights" / "manifest.json"),
        help="Path to weights manifest JSON.",
    )
    parser.add_argument(
        "--out_dir",
        default=str(Path(__file__).resolve().parents[1] / "weights"),
        help="Directory to place downloaded weights.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)

    data = json.loads(manifest_path.read_text())
    assets = data.get("assets", [])
    if not assets:
        print(f"No assets listed in {manifest_path}", file=sys.stderr)
        return 2

    base = f"https://github.com/{args.repo}/releases/download/{args.tag}"
    for asset in assets:
        name = asset["name"]
        expected = (asset.get("sha256") or "").strip().lower()
        url = f"{base}/{name}"
        dst = out_dir / name

        if dst.exists():
            if expected:
                got = _sha256(dst)
                if got == expected:
                    print(f"[ok] {name} already present (sha256 verified)")
                    continue
                print(f"[warn] {name} sha256 mismatch; re-downloading")
            else:
                print(f"[ok] {name} already present")
                continue

        print(f"[dl] {url}")
        _download(url, dst)

        if expected:
            got = _sha256(dst)
            if got != expected:
                print(f"[error] sha256 mismatch for {name}\n  expected: {expected}\n  got:      {got}", file=sys.stderr)
                return 3
            print(f"[ok] {name} (sha256 verified)")
        else:
            print(f"[ok] {name} (no sha256 provided)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


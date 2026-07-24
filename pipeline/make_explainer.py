"""Precompute rank-k reconstructions for the site's interactive
explainer: pick a city, drag k, watch the plan appear. Writes small
PNGs to docs/explainer/ and the energy shares to docs/explainer.json."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
RASTERS = ROOT / "data" / "interim" / "rasters"
OUT = ROOT / "docs" / "explainer"

KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
CITIES = [
    ("chicago", "Chicago"),
    ("barcelona", "Barcelona"),
    ("manila", "Manila"),
    ("ph-koronadal", "Koronadal"),
    ("ph-marawi", "Marawi"),
    ("fes", "Fes"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = []
    for slug, label in CITIES:
        x = np.asarray(Image.open(RASTERS / f"{slug}.png"),
                       dtype=np.float32) / 255.0
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        e = s**2
        cum = np.cumsum(e) / e.sum()
        erank = float(e.sum() ** 2 / (e**2).sum())
        energies = {}
        for k in KS:
            approx = np.clip((u[:, :k] * s[:k]) @ vt[:k], 0, 1)
            img = Image.fromarray(
                (255 - approx * 255).astype(np.uint8)).resize(
                (300, 300), Image.LANCZOS)
            img.save(OUT / f"{slug}-{k}.png", optimize=True)
            energies[str(k)] = round(float(cum[k - 1]) * 100, 1)
        img = Image.fromarray((255 - x * 255).astype(np.uint8)).resize(
            (300, 300), Image.LANCZOS)
        img.save(OUT / f"{slug}-full.png", optimize=True)
        meta.append({"slug": slug, "label": label,
                     "erank": round(erank, 1), "energy": energies})
        print(slug, "erank", round(erank, 1),
              "energy@8:", energies["8"], "@64:", energies["64"])
    (ROOT / "docs" / "explainer.json").write_text(json.dumps(
        {"ks": KS, "cities": meta}))


if __name__ == "__main__":
    main()

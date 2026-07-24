"""Frames for the site's hero mosaic: many cities assembling from
their stripe patterns, at tile resolution. Also writes
docs/hero/manifest.json with each city's label and rank."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
RASTERS = ROOT / "data" / "interim" / "rasters"
OUT = ROOT / "docs" / "hero"

KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
CITIES = [
    "chicago", "manhattan", "barcelona", "la-plata",
    "w-ar-mar-del-plata", "kyoto", "savannah", "moscow", "fes",
    "tokyo", "paris", "brasilia", "venice", "athens", "buenos-aires",
    "xian", "manila", "ph-koronadal", "ph-marawi", "ph-cebu",
    "ph-davao", "ph-baguio", "ph-vigan", "ph-quezon",
]
SIZE = 300


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {r["slug"]: r for r in csv.DictReader(
        open(ROOT / "data" / "clean" / "city_rank.csv"))}
    manifest = []
    for slug in CITIES:
        src = RASTERS / f"{slug}.png"
        if not src.exists():
            print("MISSING:", slug)
            continue
        x = np.asarray(Image.open(src), dtype=np.float32) / 255.0
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        for k in KS:
            approx = np.clip((u[:, :k] * s[:k]) @ vt[:k], 0, 1)
            img = Image.fromarray(
                (255 - approx * 255).astype(np.uint8)).resize(
                (SIZE, SIZE), Image.LANCZOS)
            img.save(OUT / f"{slug}-{k}.webp", quality=80)
        img = Image.fromarray((255 - x * 255).astype(np.uint8)).resize(
            (SIZE, SIZE), Image.LANCZOS)
        img.save(OUT / f"{slug}-full.webp", quality=80)
        r = rows.get(slug, {})
        label = r.get("name", slug).replace("City of ", "").split(" (")[0]
        manifest.append({"slug": slug, "label": label.strip(),
                         "erank": round(float(r.get("erank", 0)), 1)})
        print(slug, "done")
    (OUT / "manifest.json").write_text(json.dumps(manifest))


if __name__ == "__main__":
    main()

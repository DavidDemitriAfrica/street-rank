"""Build the GitHub Pages site data: docs/data.json, one thumbnail per
city (streets dark on white), docs/anchors.json for the synthetic
plans, and copies of the static figures."""

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joins import gdp_per_capita, join_sndi

ROOT = Path(__file__).resolve().parent.parent
RASTERS = ROOT / "data" / "interim" / "rasters"
SPECTRA = ROOT / "data" / "interim" / "spectra"
DOCS = ROOT / "docs"
THUMBS = DOCS / "thumbs"

# Sample points of the cumulative-energy curve shipped for sparklines.
CUM_KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def cum_energy(slug):
    """Share of spectral energy at CUM_KS components, as integer %."""
    f = SPECTRA / f"{slug}.npz"
    if not f.exists():
        return None
    s = np.load(f)["sigma"].astype(np.float64)
    e = s**2
    cum = np.cumsum(e) / e.sum()
    return [int(round(100 * cum[min(k, len(cum)) - 1])) for k in CUM_KS]

NUMERIC = [
    "lat", "lon", "pop2024", "pop", "netmig_rate", "rotation_deg",
    "erank", "erank_norm", "erank_l1", "k90", "erank_unrotated",
    "orient_entropy", "orient_order", "grid_share", "straightness",
    "block_m", "intersection_km2", "four_way_share", "deadend_share",
    "street_km",
]


def main():
    THUMBS.mkdir(parents=True, exist_ok=True)
    (DOCS / "assets").mkdir(exist_ok=True)
    gdp = gdp_per_capita()

    out = []
    with open(ROOT / "data" / "clean" / "city_rank.csv") as f:
        for r in csv.DictReader(f):
            if not r.get("erank") or r["slug"] == "w-ca-vancouver":
                continue
            raster = RASTERS / f"{r['slug']}.png"
            if not raster.exists():
                continue
            rec = {
                "slug": r["slug"], "name": r["name"].strip(),
                "country": r["country"], "kind": r["kind"],
                "province": r["province"], "region": r["region"],
                "income_class": r["income_class"], "flag": r["flag"],
                "gdp_pc": gdp.get(r["country"]),
            }
            for k in NUMERIC:
                v = r.get(k, "")
                try:
                    rec[k] = round(float(v), 4)
                    if rec[k] != rec[k]:  # NaN
                        rec[k] = None
                except ValueError:
                    rec[k] = None
            rec["cum"] = cum_energy(r["slug"])
            out.append(rec)

            thumb_path = THUMBS / f"{r['slug']}.png"
            if not thumb_path.exists():
                img = Image.open(raster)
                thumb = ImageOps.invert(img.convert("L")).resize(
                    (320, 320), Image.LANCZOS)
                thumb.save(thumb_path, optimize=True)

    join_sndi(out)
    (DOCS / "data.json").write_text(json.dumps(out))

    anchors = list(csv.DictReader(
        open(ROOT / "data" / "clean" / "synthetic_anchors.csv")))
    (DOCS / "anchors.json").write_text(json.dumps(anchors))

    for fig in ["rank_reconstruction.png", "sigma_decay.png",
                "region_strip.png"]:
        src = ROOT / "figures" / fig
        if src.exists():
            shutil.copy(src, DOCS / "assets" / fig)
    print(f"{len(out)} cities -> data.json; "
          f"{sum(1 for c in out if c.get('sndi') is not None)} with SNDi; "
          f"{sum(1 for c in out if c.get('gdp_pc'))} with GDP")


if __name__ == "__main__":
    main()

"""Fetch, rasterize, and measure every city; write data/clean/city_rank.csv.

Usage:
    python3 pipeline/run_all.py                 # full run
    python3 pipeline/run_all.py --only fes rome # subset by slug
    python3 pipeline/run_all.py --ph-top 80     # how many PH cities
"""

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cities import all_cities
from fetch import fetch_streets
from rank import WINDOW_M, analyze

ROOT = Path(__file__).resolve().parent.parent
RASTERS = ROOT / "data" / "interim" / "rasters"
SPECTRA = ROOT / "data" / "interim" / "spectra"
OUT = ROOT / "data" / "clean" / "city_rank.csv"

# Streets are fetched out to the window's corner radius so nothing is
# missing after rotation: 2500 * sqrt(2) ~ 3536 m, plus margin.
FETCH_RADIUS_M = 3600

FIELDS = [
    "slug", "name", "country", "kind", "province", "lat", "lon",
    "pop2024", "pop", "region", "netmig_rate", "income_class", "rotation_deg", "erank", "erank_norm", "erank_l1", "k90",
    "erank_unrotated", "orient_entropy", "orient_order", "grid_share",
    "straightness", "block_m", "intersection_km2", "four_way_share",
    "deadend_share", "street_km", "n_ways", "flag",
]


def run_city(city):
    data = fetch_streets(city["slug"], city["lat"], city["lon"],
                         FETCH_RADIUS_M)
    metrics, raster, sigma = analyze(data, city["lat"], city["lon"])
    if metrics is None:
        return {**city, "flag": (city["flag"] + ";no_streets").strip(";")}
    Image.fromarray((raster * 255).astype(np.uint8)).save(
        RASTERS / f"{city['slug']}.png"
    )
    np.savez_compressed(SPECTRA / f"{city['slug']}.npz",
                        sigma=sigma[:512].astype(np.float32))
    row = {**city, **metrics}
    if metrics["street_km"] < 40:
        row["flag"] = (row["flag"] + ";sparse_osm").strip(";")
    if metrics["street_km"] < 200:
        row["flag"] = (row["flag"] + ";norm_short").strip(";")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--ph-top", type=int, default=80)
    args = ap.parse_args()

    cities = all_cities(args.ph_top)
    if args.only:
        cities = [c for c in cities if c["slug"] in set(args.only)]
    print(f"{len(cities)} cities, window {WINDOW_M:.0f} m", flush=True)

    # Prefetch with a small worker pool (cache makes this a no-op on
    # rerun); measurement below then runs entirely off the cache.
    def prefetch(city):
        try:
            fetch_streets(city["slug"], city["lat"], city["lon"],
                          FETCH_RADIUS_M)
            print(f"  fetched {city['slug']}", flush=True)
        except Exception as e:
            print(f"  prefetch {city['slug']} failed: {e}", flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(prefetch, cities))

    rows = []
    for i, city in enumerate(cities):
        t0 = time.time()
        try:
            row = run_city(city)
        except Exception as e:
            print(f"[{i+1}/{len(cities)}] {city['slug']} FAILED: {e}",
                  flush=True)
            row = {**city, "flag": (city["flag"] + ";fetch_failed").strip(";")}
        rows.append(row)
        print(
            f"[{i+1}/{len(cities)}] {city['slug']:28s} "
            f"erank={row.get('erank', '-'):>6} "
            f"rot={row.get('rotation_deg', '-'):>5} "
            f"km={row.get('street_km', '-'):>7} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

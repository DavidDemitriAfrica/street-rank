"""Synthetic street plans, run through the same rasterizer and SVD as
real cities. They anchor the rank scale: a reader can place any city
between "perfect grid" and "random curves".

Each generator returns a list of polylines in window meters, like the
output of rank.polylines_from_overpass. Total street length is held
near 400 km, a typical big-city value for the window, so the anchors
differ in geometry rather than in amount of ink.
"""

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rank import WINDOW_M, effective_rank, rasterize, subsample_to_km

HALF = WINDOW_M / 2.0
RNG = np.random.default_rng(7)


def _grid(spacing, jitter=0.0):
    """A square grid. jitter shifts every line off its slot by up to
    jitter*spacing. Note that jitter does NOT raise the rank: full
    lines in two shared directions stay a rank-2 picture no matter how
    unevenly they are spaced. That fact is itself an anchor."""
    lines = []
    n = int(WINDOW_M // spacing)
    for i in range(n + 1):
        pos = -HALF + i * spacing
        for horizontal in (False, True):
            p = pos + RNG.uniform(-jitter, jitter) * spacing
            if horizontal:
                lines.append(np.array([[-HALF, p], [HALF, p]], dtype=np.float32))
            else:
                lines.append(np.array([[p, -HALF], [p, HALF]], dtype=np.float32))
    return lines


def _grid_broken(spacing, drop):
    """A grid whose lines have gaps: each 100 m piece survives with
    probability 1 - drop. Interruption is what raises rank."""
    lines = []
    piece = 100.0
    n = int(WINDOW_M // spacing)
    for i in range(n + 1):
        pos = -HALF + i * spacing
        for horizontal in (False, True):
            s = -HALF
            while s < HALF:
                if RNG.random() > drop:
                    seg = ([[s, pos], [min(s + piece, HALF), pos]]
                           if horizontal else
                           [[pos, s], [pos, min(s + piece, HALF)]])
                    lines.append(np.array(seg, dtype=np.float32))
                s += piece
    return lines


def _grid_staggered(spacing):
    """Full horizontal streets, but each row's cross streets are offset
    from the next row's, like brick bond. Common in suburbs."""
    lines = []
    n = int(WINDOW_M // spacing)
    for i in range(n + 1):
        y = -HALF + i * spacing
        lines.append(np.array([[-HALF, y], [HALF, y]], dtype=np.float32))
    for i in range(n):
        y0, y1 = -HALF + i * spacing, -HALF + (i + 1) * spacing
        off = RNG.uniform(0, spacing)
        x = -HALF + off
        while x < HALF:
            lines.append(np.array([[x, y0], [x, y1]], dtype=np.float32))
            x += spacing
    return lines


def _patchwork(spacing):
    """Four quadrants, each a grid rotated to its own angle. Locally
    ordered, globally misaligned, like a city grown from several towns."""
    lines = []
    for qx, qy, deg in [(-1, -1, 0), (1, -1, 20), (-1, 1, 40), (1, 1, 65)]:
        t = math.radians(deg)
        R = np.array([[math.cos(t), -math.sin(t)],
                      [math.sin(t), math.cos(t)]], dtype=np.float32)
        cx, cy = qx * HALF / 2, qy * HALF / 2
        n = int(HALF // spacing)
        for i in range(-n, n + 1):
            p = i * spacing
            for seg in ([[-HALF / 2, p], [HALF / 2, p]],
                        [[p, -HALF / 2], [p, HALF / 2]]):
                s = np.array(seg, dtype=np.float32) @ R.T
                s = s + np.array([cx, cy], dtype=np.float32)
                keep = (np.abs(s[:, 0] - cx) <= HALF / 2 + 1) & \
                       (np.abs(s[:, 1] - cy) <= HALF / 2 + 1)
                if keep.all():
                    lines.append(s)
    return lines


def _radial(n_spokes=24, ring_step=200.0):
    """Spokes and rings, the Moscow shape."""
    lines = []
    for i in range(n_spokes):
        t = 2 * math.pi * i / n_spokes
        lines.append(np.array(
            [[0, 0], [HALF * 1.5 * math.sin(t), HALF * 1.5 * math.cos(t)]],
            dtype=np.float32))
    r = ring_step
    while r < HALF * 1.5:
        ts = np.linspace(0, 2 * math.pi, 128)
        lines.append(np.column_stack(
            [r * np.sin(ts), r * np.cos(ts)]).astype(np.float32))
        r += ring_step
    return lines


def _random_chords(total_km=400.0, seg_m=400.0):
    """Straight segments thrown down at random positions and angles."""
    lines = []
    n = int(total_km * 1000 / seg_m)
    for _ in range(n):
        x, y = RNG.uniform(-HALF, HALF, 2)
        t = RNG.uniform(0, math.pi)
        dx, dy = seg_m / 2 * math.sin(t), seg_m / 2 * math.cos(t)
        lines.append(np.array([[x - dx, y - dy], [x + dx, y + dy]],
                              dtype=np.float32))
    return lines


def _random_curves(total_km=400.0, step_m=60.0):
    """Meandering walks, the fully organic extreme."""
    lines = []
    laid = 0.0
    while laid < total_km * 1000:
        x, y = RNG.uniform(-HALF, HALF, 2)
        t = RNG.uniform(0, 2 * math.pi)
        pts = [(x, y)]
        for _ in range(RNG.integers(10, 40)):
            t += RNG.normal(0, 0.5)
            x += step_m * math.sin(t)
            y += step_m * math.cos(t)
            pts.append((x, y))
            laid += step_m
        lines.append(np.array(pts, dtype=np.float32))
    return lines


ANCHORS = [
    ("grid-100m", "Perfect 100 m grid", lambda: _grid(100)),
    ("grid-uneven", "Grid, unevenly spaced", lambda: _grid(100, 0.30)),
    ("grid-staggered", "Grid with offset cross streets", lambda: _grid_staggered(100)),
    ("grid-broken", "Grid with 30% gaps", lambda: _grid_broken(100, 0.30)),
    ("patchwork", "Four grids at different angles", lambda: _patchwork(100)),
    ("radial", "Spokes and rings", _radial),
    ("random-chords", "Random straight segments, 400 km", _random_chords),
    ("curves-200km", "Random curves, 200 km", lambda: _random_curves(200)),
    ("curves-400km", "Random curves, 400 km", lambda: _random_curves(400)),
    ("curves-800km", "Random curves, 800 km", lambda: _random_curves(800)),
]


def main():
    root = Path(__file__).resolve().parent.parent
    outdir = root / "docs" / "anchors"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for slug, label, gen in ANCHORS:
        lines = gen()
        raster = rasterize(lines)
        er, l1, k90, _ = effective_rank(raster)
        er_norm, _, _, _ = effective_rank(
            rasterize(subsample_to_km(lines)))
        rows.append((slug, label, round(er, 1), round(er_norm, 1), k90))
        img = Image.fromarray(
            (255 - raster * 255).astype(np.uint8)).resize((256, 256))
        img.save(outdir / f"{slug}.png", optimize=True)
        print(f"{slug:16s} {label:34s} erank={er:6.1f} "
              f"norm={er_norm:6.1f} k90={k90}")
    import csv
    with open(root / "data" / "clean" / "synthetic_anchors.csv", "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["slug", "label", "erank", "erank_norm", "k90"])
        w.writerows(rows)


if __name__ == "__main__":
    main()

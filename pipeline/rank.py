"""Rasterize a city's street network at fixed scale and measure its
effective rank.

The window is WINDOW_M x WINDOW_M meters drawn onto an N x N binary image
(one pixel per ~4.9 m). Before drawing, the street coordinates are rotated
so the dominant bearing (mod 90 degrees) is axis-aligned; without this a
45-degree-rotated grid would look high-rank to an axis-aligned SVD.

Effective rank follows Roy & Vetterli (2007): exp of the Shannon entropy
of the L1-normalized singular values. A perfect grid is ~rank 2 (avenues
are one outer product, cross streets another); organic fabric needs
hundreds of components.
"""

import math

import numpy as np
from PIL import Image, ImageDraw

WINDOW_M = 5000.0
N_PX = 1024


def polylines_from_overpass(data, lat0, lon0):
    """Way geometries -> list of float32 arrays of local (x, y) meters."""
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 111320.0
    lines = []
    for el in data.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        pts = np.array(
            [((p["lon"] - lon0) * kx, (p["lat"] - lat0) * ky)
             for p in el["geometry"]],
            dtype=np.float32,
        )
        if len(pts) >= 2:
            lines.append(pts)
    return lines


def _segments(lines):
    """Stack all polyline segments: returns (P0, P1, lengths, bearings_deg).

    Bearings are undirected, in [0, 180), measured clockwise from north.
    """
    a = np.concatenate([ln[:-1] for ln in lines])
    b = np.concatenate([ln[1:] for ln in lines])
    d = b - a
    lengths = np.hypot(d[:, 0], d[:, 1])
    keep = lengths > 0.5
    a, b, d, lengths = a[keep], b[keep], d[keep], lengths[keep]
    bearings = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 180.0
    return a, b, lengths, bearings


def dominant_bearing(lines):
    """Length-weighted modal bearing mod 90, via a smoothed 1-degree
    circular histogram. Returns degrees in [0, 90)."""
    _, _, lengths, bearings = _segments(lines)
    hist, _ = np.histogram(bearings % 90.0, bins=90, range=(0, 90),
                           weights=lengths)
    # circular smoothing over +/- 2 degrees
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    smooth = np.convolve(np.tile(hist, 3), kernel, mode="same")[90:180]
    return float(np.argmax(smooth)) + 0.5


def orientation_entropy(lines):
    """Boeing (2019)-style street orientation entropy in nats:
    36 ten-degree bins over 0-360, each undirected segment counted in both
    directions, length-weighted. Max is ln(36) ~ 3.58."""
    _, _, lengths, bearings = _segments(lines)
    both = np.concatenate([bearings, bearings + 180.0]) % 360.0
    w = np.concatenate([lengths, lengths])
    hist, _ = np.histogram(both, bins=36, range=(0, 360), weights=w)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _rotate(lines, deg):
    """Rotate xy coords CCW by deg, mapping bearing-deg directions north."""
    t = math.radians(deg)
    R = np.array([[math.cos(t), -math.sin(t)],
                  [math.sin(t), math.cos(t)]], dtype=np.float32)
    return [ln @ R.T for ln in lines]


def rasterize(lines):
    """Draw polylines (local meters) into an N_PX x N_PX binary image."""
    img = Image.new("L", (N_PX, N_PX), 0)
    draw = ImageDraw.Draw(img)
    scale = N_PX / WINDOW_M
    half = WINDOW_M / 2.0
    m = half + 200
    for ln in lines:
        # skip polylines strictly on one side of the window (with margin)
        if ((ln[:, 0] > m).all() or (ln[:, 0] < -m).all()
                or (ln[:, 1] > m).all() or (ln[:, 1] < -m).all()):
            continue
        px = (ln[:, 0] + half) * scale
        py = (half - ln[:, 1]) * scale
        draw.line(list(zip(px.tolist(), py.tolist())), fill=255, width=1)
    return np.asarray(img, dtype=np.float32) / 255.0


def effective_rank(a):
    """Effective ranks of a raster. Returns (energy PR, L1 erank, k90,
    sigma).

    The primary measure is the participation ratio of the squared
    singular values, (sum l)^2 / sum l^2 with l = sigma^2 — it measures
    how concentrated the image's energy is in its top components and
    separates planned from organic fabric far better than the
    L1/entropy variant, which saturates on thin-line rasters. The
    Roy & Vetterli (2007) entropy erank is kept as a secondary column,
    and k90 is the number of components holding 90% of the energy."""
    s = np.linalg.svd(a, compute_uv=False)
    s = s[s > 1e-9]
    if len(s) == 0:
        return 0.0, 0.0, 0, s
    e = s**2
    pr = float(e.sum() ** 2 / (e**2).sum())
    p = s / s.sum()
    l1 = float(np.exp(-(p * np.log(p)).sum()))
    frac = np.cumsum(e) / e.sum()
    k90 = int(np.searchsorted(frac, 0.9) + 1)
    return pr, l1, k90, s


def orientation_order(entropy_nats):
    """Boeing (2019)'s phi: 1 for a perfect 4-bearing grid, 0 for a
    uniform bearing distribution."""
    h_grid, h_max = math.log(4), math.log(36)
    phi = 1.0 - ((entropy_nats - h_grid) / (h_max - h_grid)) ** 2
    return float(min(max(phi, 0.0), 1.0))


def grid_share(lines, rot):
    """Fraction of in-window street length within 10 degrees (mod 90)
    of the dominant axes."""
    a, b, lengths, bearings = _segments(lines)
    mid = (a + b) / 2.0
    half = WINDOW_M / 2.0
    inside = (np.abs(mid[:, 0]) <= half) & (np.abs(mid[:, 1]) <= half)
    if not inside.any():
        return 0.0
    d = np.abs((bearings[inside] - rot) % 90.0)
    d = np.minimum(d, 90.0 - d)
    w = lengths[inside]
    return float(w[d <= 10.0].sum() / w.sum())


def straightness(data, lat0, lon0):
    """Length-weighted mean chord/path ratio per way (1 = dead straight).
    Loops (roundabouts) and stubs under 50 m are skipped."""
    kx = 111320.0 * math.cos(math.radians(lat0))
    num = den = 0.0
    for el in data.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        g = el["geometry"]
        pts = np.array([((p["lon"] - lon0) * kx,
                         (p["lat"] - lat0) * 111320.0) for p in g],
                       dtype=np.float64)
        if len(pts) < 2:
            continue
        seg = np.diff(pts, axis=0)
        path = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
        chord = float(np.hypot(*(pts[-1] - pts[0])))
        mid = pts.mean(axis=0)
        if path < 50 or chord < 1 or np.abs(mid).max() > WINDOW_M / 2:
            continue
        num += min(chord / path, 1.0) * path
        den += path
    return float(num / den) if den else float("nan")


def node_stats(data, lat0, lon0):
    """Intersection density and junction mix from way topology.

    A node's degree counts one edge per way end and two per interior
    pass-through. Junctions are degree >= 3, dead ends degree 1 (curve
    points along a single way are degree 2 and count as neither)."""
    kx = 111320.0 * math.cos(math.radians(lat0))
    degree, coord = {}, {}
    for el in data.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        ids, geom = el.get("nodes", []), el["geometry"]
        if len(ids) != len(geom):
            continue
        for i, nid in enumerate(ids):
            inc = 1 if i in (0, len(ids) - 1) else 2
            degree[nid] = degree.get(nid, 0) + inc
            if nid not in coord:
                coord[nid] = ((geom[i]["lon"] - lon0) * kx,
                              (geom[i]["lat"] - lat0) * 111320.0)
    half = WINDOW_M / 2.0
    j = d1 = d4 = 0
    for nid, deg in degree.items():
        x, y = coord[nid]
        if abs(x) > half or abs(y) > half:
            continue
        if deg >= 3:
            j += 1
            if deg >= 4:
                d4 += 1
        elif deg == 1:
            d1 += 1
    area_km2 = (WINDOW_M / 1000.0) ** 2
    return {
        "intersection_km2": round(j / area_km2, 1),
        "four_way_share": round(d4 / j, 3) if j else float("nan"),
        "deadend_share": round(d1 / (d1 + j), 3) if d1 + j else float("nan"),
    }


def block_spacing_m(raster):
    """Dominant street spacing in meters, from the FFT of the aligned
    raster's row and column density profiles. Returns (spacing, peak
    strength as multiple of median band power); spacing is NaN when no
    clear periodicity exists."""
    m_per_px = WINDOW_M / N_PX
    best = (float("nan"), 0.0)
    for axis in (0, 1):
        prof = raster.sum(axis=axis)
        prof = prof - prof.mean()
        power = np.abs(np.fft.rfft(prof)) ** 2
        freqs = np.fft.rfftfreq(len(prof))  # cycles per pixel
        with np.errstate(divide="ignore"):
            spacing = m_per_px / freqs
        band = (spacing >= 60) & (spacing <= 400)
        if not band.any():
            continue
        i = np.argmax(power * band)
        strength = float(power[i] / (np.median(power[band]) + 1e-9))
        if strength > best[1]:
            best = (float(spacing[i]), strength)
    spacing, strength = best
    if strength < 8.0:
        return float("nan"), strength
    return round(spacing, 0), strength


def subsample_to_km(lines, target_km=200.0, seed=0):
    """Keep a random subset of whole streets totalling ~target_km.

    The plain effective rank depends on how much street there is, not
    only on its shape: past a point, adding ink concentrates spectral
    energy and the rank falls even for disordered fabric. Holding total
    length fixed removes that. Whole polylines are kept or dropped so
    street shapes stay intact."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lines))
    kept, total = [], 0.0
    for i in order:
        ln = lines[i]
        seg = np.diff(ln, axis=0)
        total += float(np.hypot(seg[:, 0], seg[:, 1]).sum())
        kept.append(lines[i])
        if total >= target_km * 1000:
            break
    return kept


def street_km_in_window(lines):
    a, b, lengths, _ = _segments(lines)
    mid = (a + b) / 2.0
    half = WINDOW_M / 2.0
    inside = (np.abs(mid[:, 0]) <= half) & (np.abs(mid[:, 1]) <= half)
    return float(lengths[inside].sum() / 1000.0)


def analyze(data, lat0, lon0):
    """Full per-city measurement. Returns (metrics dict, aligned raster,
    singular values of the aligned raster)."""
    lines = polylines_from_overpass(data, lat0, lon0)
    if not lines:
        return None, None, None
    rot = dominant_bearing(lines)
    aligned = _rotate(lines, rot)

    raster = rasterize(aligned)
    erank, erank_l1, k90, sigma = effective_rank(raster)
    erank_unrot, _, _, _ = effective_rank(rasterize(lines))
    erank_norm, _, _, _ = effective_rank(
        rasterize(subsample_to_km(aligned)))
    entropy = orientation_entropy(lines)
    block_m, block_strength = block_spacing_m(raster)

    metrics = {
        "rotation_deg": round(rot, 1),
        "erank": round(erank, 1),
        "erank_norm": round(erank_norm, 1),
        "erank_l1": round(erank_l1, 1),
        "k90": k90,
        "erank_unrotated": round(erank_unrot, 1),
        "orient_entropy": round(entropy, 3),
        "orient_order": round(orientation_order(entropy), 3),
        "grid_share": round(grid_share(aligned, 0.0), 3),
        "straightness": round(straightness(data, lat0, lon0), 3),
        "block_m": block_m,
        "street_km": round(street_km_in_window(lines), 1),
        "n_ways": sum(
            1 for el in data["elements"] if el.get("type") == "way"
        ),
        **node_stats(data, lat0, lon0),
    }
    return metrics, raster, sigma

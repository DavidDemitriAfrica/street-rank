"""All figures. Reads data/clean/city_rank.csv plus the cached rasters
and spectra; writes PNGs to figures/.

Style follows the philippines-internal-migration repo: serif type, white
surface, navy + coral accents, recessive grid, direct labels on every
colored series (coral sits below 3:1 contrast on white, so nothing is
identified by color alone).
"""

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle
from figstyle import CORAL, GRAY_DEEMPH, GRID, INK, INK2, NAVY

ROOT = Path(__file__).resolve().parent.parent
RASTERS = ROOT / "data" / "interim" / "rasters"
SPECTRA = ROOT / "data" / "interim" / "spectra"
FIGS = ROOT / "figures"

figstyle.apply()


def load_rows():
    with open(ROOT / "data" / "clean" / "city_rank.csv") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("erank") and r["slug"] != "w-ca-vancouver"]
    numeric = ["erank", "erank_norm", "orient_entropy", "street_km",
               "netmig_rate", "pop2024", "pop"]
    for r in rows:
        for k in numeric:
            try:
                r[k] = float(r.get(k, ""))
            except ValueError:
                r[k] = None
    return rows


def short_name(r):
    n = r["name"]
    for junk in ["City of ", " (core)", " (Eixample)", " (Medina)", " (Deira)"]:
        n = n.replace(junk, "")
    return n.split(" (")[0]


def _tidy(ax, xlab, ylab):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c9c8c2")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)


def gallery(rows, path, ncols=8, cell=200):
    """Street rasters, black on white, sorted by effective rank."""
    rows = [r for r in rows if (RASTERS / f"{r['slug']}.png").exists()]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["erank"])
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 1.75, nrows * 2.02)
    )
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, r in zip(np.ravel(axes), rows):
        img = Image.open(RASTERS / f"{r['slug']}.png").resize(
            (cell, cell), Image.LANCZOS
        )
        a = 1.0 - np.asarray(img, dtype=np.float32) / 255.0
        ax.imshow(a, cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"{short_name(r)}\n{r['erank']:.0f}", fontsize=7.5, pad=2,
            color=INK,
        )
    fig.suptitle(
        "Effective rank of the street plan, 5 km × 5 km around the "
        "center (lower = more grid-like)",
        y=1.0, fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def sigma_decay(rows):
    """Cumulative spectral energy vs number of components."""
    highlights = {
        "chicago": (NAVY, "Chicago"),
        "fes": (CORAL, "Fes"),
        "ph-quezon-city": (INK2, "Quezon City"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r in rows:
        f = SPECTRA / f"{r['slug']}.npz"
        if not f.exists():
            continue
        s = np.load(f)["sigma"].astype(np.float64)
        e = s**2
        cum = np.cumsum(e) / e.sum()
        k = np.arange(1, len(cum) + 1)
        if r["slug"] in highlights:
            color, label = highlights[r["slug"]]
            ax.plot(k, cum, color=color, linewidth=2.0, zorder=3)
            i = int(np.searchsorted(cum, 0.9))
            ax.annotate(
                label, (k[i], cum[i]), xytext=(6, -10),
                textcoords="offset points", fontsize=9, color=color,
            )
        else:
            ax.plot(k, cum, color=GRAY_DEEMPH, linewidth=0.5, alpha=0.35,
                    zorder=1)
    ax.axhline(0.9, color=GRID, linewidth=0.9, zorder=0)
    ax.text(1.1, 0.905, "90% of energy", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(1, 512)
    ax.set_ylim(0, 1.02)
    _tidy(ax, "components kept", "share of spectral energy")
    ax.set_title(
        "How many singular vectors does it take to draw a city?",
        fontsize=11, loc="left",
    )
    fig.tight_layout()
    fig.savefig(FIGS / "sigma_decay.png", dpi=180)
    plt.close(fig)


def reconstruction_strip():
    """Rank-k reconstructions: a gridded city vs an organic one."""
    ks = [2, 8, 32, 128]
    picks = [("chicago", "Chicago"), ("fes", "Fes")]
    fig, axes = plt.subplots(
        2, len(ks) + 1, figsize=(2.1 * (len(ks) + 1), 4.5)
    )
    for row, (slug, label) in enumerate(picks):
        x = np.asarray(
            Image.open(RASTERS / f"{slug}.png"), dtype=np.float32
        ) / 255.0
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        for col, k in enumerate(ks):
            approx = np.clip((u[:, :k] * s[:k]) @ vt[:k], 0, 1)
            ax = axes[row, col]
            ax.imshow(approx, cmap="gray_r", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"rank {k}", fontsize=10, color=INK2)
        ax = axes[row, len(ks)]
        ax.imshow(x, cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")
        if row == 0:
            ax.set_title("full", fontsize=10, color=INK2)
        axes[row, 0].set_ylabel(label)
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "A gridded city compresses; an organic one does not", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGS / "rank_reconstruction.png", dpi=170)
    plt.close(fig)


def scatter_entropy(rows):
    """Effective rank vs Boeing-style orientation entropy."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for kind, color, label in [
        ("global", NAVY, "world cities"),
        ("ph", CORAL, "Philippine cities"),
    ]:
        pts = [r for r in rows if r["kind"] == kind]
        ax.scatter(
            [r["orient_entropy"] for r in pts],
            [r["erank"] for r in pts],
            s=18, color=color, alpha=0.75, linewidths=0, label=label,
        )
    labels = {"chicago": (5, 3), "barcelona": (5, 3), "fes": (5, 3),
              "moscow": (5, 3), "manila": (-14, -13), "tokyo": (6, -4),
              "ph-vigan": (-38, 2), "ph-marawi": (-46, -3),
              "ph-koronadal": (6, -4)}
    for slug, off in labels.items():
        r = next((x for x in rows if x["slug"] == slug), None)
        if r:
            ax.annotate(
                short_name(r), (r["orient_entropy"], r["erank"]),
                xytext=off, textcoords="offset points", fontsize=8,
                color=INK,
            )
    ax.set_yscale("log")
    ax.set_yticks([15, 25, 50, 100, 200])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    _tidy(ax, "street orientation entropy (nats, max 3.58)",
          "effective rank (log scale)")
    ax.set_title(
        "Effective rank sees more than orientation entropy",
        fontsize=11, loc="left",
    )
    figstyle.box_legend(ax.legend(loc="upper left", fontsize=9))
    fig.tight_layout()
    fig.savefig(FIGS / "scatter_entropy.png", dpi=180)
    plt.close(fig)


def scatter_pop_ph(rows):
    """PH only: does plan complexity scale with population?"""
    pts = [r for r in rows if r["kind"] == "ph" and r.get("pop2024")]
    if len(pts) < 3:
        return
    x = np.array([float(r["pop2024"]) for r in pts])
    y = np.array([r["erank"] for r in pts])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(x, y, s=20, color=CORAL, alpha=0.8, linewidths=0)
    b, a = np.polyfit(np.log10(x), np.log10(y), 1)
    xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
    ax.plot(xs, 10**a * xs**b, color=NAVY, linewidth=1.6)
    ax.text(
        0.03, 0.04, f"log-log slope = {b:.2f} — essentially flat",
        transform=ax.transAxes, fontsize=9, color=NAVY,
    )
    labels = {"ph-quezon-city": (6, 3), "ph-davao": (6, 8),
              "ph-manila": (2, -13), "ph-vigan": (6, 3),
              "ph-cebu": (6, 3), "ph-baguio": (6, 3),
              "ph-marawi": (6, 3), "ph-koronadal": (6, 3)}
    for r in pts:
        if r["slug"] in labels:
            ax.annotate(
                short_name(r), (float(r["pop2024"]), r["erank"]),
                xytext=labels[r["slug"]], textcoords="offset points",
                fontsize=8, color=INK,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([50, 100, 200])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    _tidy(ax, "municipal population, 2024 census",
          "effective rank (log scale)")
    ax.set_title(
        "Street-plan complexity vs city size, 80 Philippine cities",
        fontsize=11, loc="left",
    )
    fig.tight_layout()
    fig.savefig(FIGS / "scatter_pop_ph.png", dpi=180)
    plt.close(fig)


def region_strip(rows):
    """Normalized rank by world region, one jittered dot per city, the
    Philippines as its own highlighted row."""
    rng = np.random.default_rng(3)
    groups = {}
    for r in rows:
        if r["erank"] is None:
            continue
        key = "Philippines" if r["kind"] == "ph" else (r["region"] or "?")
        groups.setdefault(key, []).append(r["erank"])
    order = sorted(groups, key=lambda g: np.median(groups[g]))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, g in enumerate(order):
        vals = np.array(groups[g])
        y = i + rng.uniform(-0.16, 0.16, len(vals))
        color = CORAL if g == "Philippines" else NAVY
        ax.scatter(vals, y, s=12, color=color, alpha=0.55, linewidths=0)
        med = np.median(vals)
        ax.plot([med, med], [i - 0.28, i + 0.28], color=INK, linewidth=1.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(
        [f"{g}  ({len(groups[g])})" for g in order], fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_xlabel("effective rank (low = grid-like)")
    ax.set_title("Street-plan order by world region, medians marked",
                 fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "region_strip.png", dpi=180)
    plt.close(fig)


def density_panels(rows):
    """The density confound, before and after normalization."""
    pts = [r for r in rows if r["erank_norm"] is not None
           and r["street_km"] and r["street_km"] >= 200]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), sharex=True)
    for ax, key, label in [(axes[0], "erank", "effective rank, raw"),
                           (axes[1], "erank_norm",
                            "effective rank at equal street length")]:
        x = np.array([r["street_km"] for r in pts])
        y = np.array([r[key] for r in pts])
        ax.scatter(x, y, s=14, color=NAVY, alpha=0.55, linewidths=0)
        rho = np.corrcoef(x, np.log(y))[0, 1]
        ax.text(0.03, 0.93, f"corr with log rank: {rho:+.2f}",
                transform=ax.transAxes, fontsize=9, color=INK2)
        _tidy(ax, "street km in the window", label)
    fig.suptitle("Rank against street density, raw and at fixed "
                 "street length (each version has a caveat)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "density_panels.png", dpi=180)
    plt.close(fig)


def sndi_scatter(rows):
    """Our geometry measure against the street-network sprawl index."""
    from joins import join_sndi
    pts = join_sndi([dict(r) for r in rows])
    pts = [r for r in pts if r.get("sndi") is not None
           and r["erank"] is not None]
    if len(pts) < 10:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x = np.array([r["sndi"] for r in pts])
    y = np.array([r["erank"] for r in pts])
    ax.scatter(x, y, s=18, color=NAVY, alpha=0.7, linewidths=0)
    from scipy.stats import spearmanr
    rho = spearmanr(x, y).statistic
    ax.text(0.03, 0.93, f"Spearman {rho:+.2f}, n = {len(pts)}",
            transform=ax.transAxes, fontsize=9, color=INK2)
    for slug, off in [("chicago", (5, 3)), ("fes", (5, 3)),
                      ("tokyo", (5, 3)), ("manila", (5, 3))]:
        r = next((p for p in pts if p["slug"] == slug), None)
        if r:
            ax.annotate(short_name(r), (r["sndi"], r["erank"]),
                        xytext=off, textcoords="offset points",
                        fontsize=8, color=INK)
    _tidy(ax, "street-network sprawl index (SNDi, connectivity)",
          "effective rank (geometry)")
    ax.set_title("Two different instruments, related verdicts",
                 fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "sndi_scatter.png", dpi=180)
    plt.close(fig)


def main():
    rows = load_rows()
    ok = [r for r in rows if "sparse_osm" not in r.get("flag", "")
          and "fetch_failed" not in r.get("flag", "")]
    gallery([r for r in ok if r["kind"] == "global"],
            FIGS / "gallery_global.png")
    gallery([r for r in ok if r["kind"] == "ph"], FIGS / "gallery_ph.png")
    sigma_decay(ok)
    reconstruction_strip()
    scatter_entropy(ok)
    scatter_pop_ph(ok)
    region_strip(ok)
    density_panels(ok)
    sndi_scatter(ok)
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()

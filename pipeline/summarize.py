"""Print every number the README and site copy quote, from the current
CSV. Run after run_all.py so the writeup never drifts from the data."""

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joins import gdp_per_capita, join_sndi

ROOT = Path(__file__).resolve().parent.parent


def main():
    rows = [r for r in csv.DictReader(
        open(ROOT / "data" / "clean" / "city_rank.csv")) if r.get("erank")]
    for r in rows:
        for k in ["erank", "erank_norm", "orient_order", "street_km",
                  "netmig_rate", "pop2024", "pop"]:
            try:
                r[k] = float(r[k])
            except (ValueError, KeyError):
                r[k] = None
    gdp = gdp_per_capita()
    for r in rows:
        r["gdp_pc"] = gdp.get(r["country"])
        r["popAny"] = r["pop2024"] or r["pop"]
    join_sndi(rows)

    n = len(rows)
    ph = [r for r in rows if r["kind"] == "ph"]
    world = [r for r in rows if r["kind"] != "ph"]
    print(f"n={n} (world {len(world)}, ph {len(ph)})")
    print(f"flags: sparse={sum('sparse_osm' in r['flag'] for r in rows)}, "
          f"norm_short={sum('norm_short' in r['flag'] for r in rows)}, "
          f"failed={sum('fetch_failed' in r['flag'] for r in rows)}")
    print(f"total street km: {sum(r['street_km'] or 0 for r in rows):,.0f}")

    for label, grp in [("WORLD", world), ("PH", ph)]:
        s = sorted((r for r in grp if r["erank_norm"]),
                   key=lambda r: r["erank_norm"])
        print(f"\n{label} most grid-like:")
        for r in s[:10]:
            print(f"  {r['name'][:30]:32s} {r['country'][:12]:13s} "
                  f"norm={r['erank_norm']:6.1f} raw={r['erank']:6.1f}")
        print(f"{label} most organic:")
        for r in s[-10:]:
            print(f"  {r['name'][:30]:32s} {r['country'][:12]:13s} "
                  f"norm={r['erank_norm']:6.1f} raw={r['erank']:6.1f}")

    print("\nregion medians (erank_norm):")
    groups = {}
    for r in rows:
        key = "Philippines" if r["kind"] == "ph" else r["region"]
        if r["erank_norm"]:
            groups.setdefault(key, []).append(r["erank_norm"])
    for g in sorted(groups, key=lambda g: np.median(groups[g])):
        print(f"  {g:15s} {np.median(groups[g]):6.1f}  (n={len(groups[g])})")

    def rho(xk, yk="erank_norm", sub=rows):
        pts = [(r[xk], r[yk]) for r in sub
               if r.get(xk) is not None and r.get(yk) is not None]
        if len(pts) < 10:
            return None, 0
        x, y = zip(*pts)
        return spearmanr(x, y).statistic, len(pts)

    print("\nspearman with erank_norm:")
    for xk in ["orient_order", "sndi", "gdp_pc", "popAny", "street_km"]:
        v, m = rho(xk)
        print(f"  {xk:15s} {v:+.2f} (n={m})" if v is not None else f"  {xk}: n/a")
    v, m = rho("netmig_rate", sub=ph)
    print(f"  PH netmig_rate  {v:+.2f} (n={m})" if v is not None else "  netmig n/a")
    v, m = rho("popAny", sub=ph)
    print(f"  PH popAny       {v:+.2f} (n={m})")

    for slug in ["manhattan", "barcelona", "xian", "chicago"]:
        r = next((x for x in rows if x["slug"] == slug), None)
        if r:
            print(f"block {slug}: {r['block_m']}")


if __name__ == "__main__":
    main()

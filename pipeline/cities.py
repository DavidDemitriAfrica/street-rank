"""City lists.

GLOBAL: hand-picked world cities spanning the planned-grid to organic
spectrum, with city-center coordinates (the classic core, not the
administrative centroid).

Philippine cities: the top municipalities by 2024 population from the
philippines-internal-migration dataset, located at their OSM place node
(the poblacion), disambiguated by proximity to the municipality polygon.
"""

import csv
import json
import unicodedata
from pathlib import Path

from fetch import fetch_ph_places

MIGRATION_REPO = Path("/home/ubuntu/migration-dataset")

# slug, name, country, lat, lon
GLOBAL = [
    # North America
    ("chicago", "Chicago", "United States", 41.878, -87.630),
    ("manhattan", "Manhattan (New York)", "United States", 40.758, -73.985),
    ("philadelphia", "Philadelphia", "United States", 39.952, -75.165),
    ("washington-dc", "Washington, D.C.", "United States", 38.905, -77.036),
    ("savannah", "Savannah", "United States", 32.076, -81.093),
    ("salt-lake-city", "Salt Lake City", "United States", 40.763, -111.888),
    ("portland", "Portland", "United States", 45.520, -122.676),
    ("phoenix", "Phoenix", "United States", 33.451, -112.073),
    ("houston", "Houston", "United States", 29.758, -95.365),
    ("boston", "Boston", "United States", 42.357, -71.059),
    ("san-francisco", "San Francisco", "United States", 37.780, -122.417),
    ("los-angeles", "Los Angeles", "United States", 34.048, -118.251),
    ("toronto", "Toronto", "Canada", 43.653, -79.383),
    ("mexico-city", "Mexico City", "Mexico", 19.433, -99.140),
    # South America
    ("la-plata", "La Plata", "Argentina", -34.921, -57.954),
    ("buenos-aires", "Buenos Aires", "Argentina", -34.606, -58.437),
    ("brasilia", "Brasilia", "Brazil", -15.794, -47.882),
    ("sao-paulo", "Sao Paulo", "Brazil", -23.550, -46.634),
    ("rio-de-janeiro", "Rio de Janeiro", "Brazil", -22.906, -43.183),
    ("bogota", "Bogota", "Colombia", 4.601, -74.077),
    ("lima", "Lima", "Peru", -12.046, -77.031),
    ("santiago", "Santiago", "Chile", -33.437, -70.650),
    # Europe
    ("barcelona", "Barcelona (Eixample)", "Spain", 41.393, 2.164),
    ("madrid", "Madrid", "Spain", 40.417, -3.703),
    ("seville", "Seville", "Spain", 37.390, -5.994),
    ("lisbon", "Lisbon", "Portugal", 38.713, -9.139),
    ("paris", "Paris", "France", 48.857, 2.347),
    ("london", "London", "United Kingdom", 51.514, -0.106),
    ("amsterdam", "Amsterdam", "Netherlands", 52.370, 4.895),
    ("berlin", "Berlin", "Germany", 52.520, 13.405),
    ("mannheim", "Mannheim", "Germany", 49.489, 8.467),
    ("vienna", "Vienna", "Austria", 48.209, 16.372),
    ("prague", "Prague", "Czechia", 50.087, 14.420),
    ("rome", "Rome", "Italy", 41.897, 12.482),
    ("turin", "Turin", "Italy", 45.071, 7.686),
    ("venice", "Venice", "Italy", 45.438, 12.335),
    ("athens", "Athens", "Greece", 37.984, 23.728),
    ("moscow", "Moscow", "Russia", 55.755, 37.617),
    ("st-petersburg", "Saint Petersburg", "Russia", 59.935, 30.325),
    # Middle East & Africa
    ("istanbul", "Istanbul", "Turkey", 41.013, 28.955),
    ("tehran", "Tehran", "Iran", 35.700, 51.420),
    ("tel-aviv", "Tel Aviv", "Israel", 32.080, 34.780),
    ("dubai", "Dubai (Deira)", "United Arab Emirates", 25.265, 55.310),
    ("cairo", "Cairo", "Egypt", 30.045, 31.240),
    ("tunis", "Tunis (Medina)", "Tunisia", 36.798, 10.171),
    ("fes", "Fes (Medina)", "Morocco", 34.061, -4.978),
    ("marrakesh", "Marrakesh", "Morocco", 31.629, -7.987),
    ("casablanca", "Casablanca", "Morocco", 33.595, -7.619),
    ("lagos", "Lagos", "Nigeria", 6.455, 3.390),
    ("nairobi", "Nairobi", "Kenya", -1.284, 36.823),
    ("johannesburg", "Johannesburg", "South Africa", -26.204, 28.042),
    # Asia & Oceania
    ("delhi", "Delhi", "India", 28.644, 77.216),
    ("mumbai", "Mumbai", "India", 18.940, 72.835),
    ("chandigarh", "Chandigarh", "India", 30.741, 76.782),
    ("islamabad", "Islamabad", "Pakistan", 33.710, 73.055),
    ("beijing", "Beijing", "China", 39.906, 116.391),
    ("shanghai", "Shanghai", "China", 31.230, 121.470),
    ("xian", "Xi'an", "China", 34.261, 108.942),
    ("kyoto", "Kyoto", "Japan", 35.011, 135.768),
    ("tokyo", "Tokyo", "Japan", 35.680, 139.767),
    ("osaka", "Osaka", "Japan", 34.686, 135.510),
    ("seoul", "Seoul", "South Korea", 37.570, 126.980),
    ("bangkok", "Bangkok", "Thailand", 13.750, 100.516),
    ("hanoi", "Hanoi", "Vietnam", 21.028, 105.852),
    ("singapore", "Singapore", "Singapore", 1.300, 103.850),
    ("jakarta", "Jakarta", "Indonesia", -6.175, 106.827),
    ("manila", "Manila (core)", "Philippines", 14.600, 120.980),
    ("sydney", "Sydney", "Australia", -33.870, 151.207),
    ("melbourne", "Melbourne", "Australia", -37.814, 144.963),
    ("adelaide", "Adelaide", "Australia", -34.926, 138.600),
    ("canberra", "Canberra", "Australia", -35.281, 149.128),
]

# Included regardless of population rank (notable street plans).
PH_ALWAYS = ["City of Vigan"]

CONTINENT_NAMES = {"AF": "Africa", "AS": "Asia", "EU": "Europe",
                   "NA": "North America", "SA": "South America",
                   "OC": "Oceania", "AN": "Antarctica"}
COUNTRY_ALIASES = {"Russia": "Russian Federation", "South Korea": "South Korea",
                   "Czechia": "Czechia"}


def _country_info():
    """iso2 -> (country name, continent name) from GeoNames."""
    out = {}
    path = Path(__file__).resolve().parent.parent / "data" / "raw" / "countryInfo.txt"
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        out[f[0]] = (f[4], CONTINENT_NAMES.get(f[8], ""))
    return out


# Most populous cities per continent, so Europe is not crowded out by
# Asian megacities. Roughly proportional to urban population but with a
# floor for the small continents. A quota that runs out of >=200k cities
# simply under-fills.
WORLD_QUOTAS = {"AS": 240, "AF": 150, "EU": 150, "NA": 110, "SA": 100,
                "OC": 30}


def world_top(n=None):
    """The most populous cities per continent from GeoNames cities15000,
    excluding the Philippines (handled with census data) and any city
    within ~17 km of a hand-picked one (the curated coordinates win)."""
    info = _country_info()
    continent_of = {}
    path = Path(__file__).resolve().parent.parent / "data" / "raw" / "countryInfo.txt"
    for line in path.read_text().splitlines():
        if not line.startswith("#") and line.strip():
            f = line.split("\t")
            continent_of[f[0]] = f[8]
    path = Path(__file__).resolve().parent.parent / "data" / "raw" / "cities15000.txt"
    rows = []
    for line in path.read_text().splitlines():
        f = line.split("\t")
        if len(f) < 15 or f[6] != "P" or f[8] == "PH":
            continue
        try:
            pop = int(f[14])
        except ValueError:
            continue
        if pop >= 200000:
            rows.append((f[2], float(f[4]), float(f[5]), f[8], pop))
    rows.sort(key=lambda r: -r[4])

    curated = [(lat, lon) for _, _, _, lat, lon in GLOBAL]
    # Manual exclusions: windows that are mostly water or parkland and
    # produce misleading ranks.
    excluded = {"w-ca-vancouver"}
    taken = {k: 0 for k in WORLD_QUOTAS}
    picked, out = [], []
    for name, lat, lon, iso2, pop in rows:
        cont = continent_of.get(iso2, "")
        if cont not in WORLD_QUOTAS or taken[cont] >= WORLD_QUOTAS[cont]:
            continue
        if any((lat - a) ** 2 + (lon - b) ** 2 < 0.023 for a, b in curated):
            continue
        if any((lat - a) ** 2 + (lon - b) ** 2 < 0.0064 for a, b in picked):
            continue
        picked.append((lat, lon))
        taken[cont] += 1
        country, region = info.get(iso2, (iso2, ""))
        slug = "w-" + iso2.lower() + "-" + "".join(
            c if c.isalnum() else "-" for c in name.lower()
        ).strip("-")
        if slug in excluded:
            continue
        seen = {c["slug"] for c in out}
        base, k = slug, 2
        while slug in seen:
            slug = f"{base}-{k}"
            k += 1
        out.append({
            "slug": slug, "name": name, "country": country,
            "kind": "world", "lat": round(lat, 5), "lon": round(lon, 5),
            "province": "", "pop2024": "", "pop": pop, "region": region,
            "netmig_rate": "", "income_class": "", "flag": "",
        })
    return out


def _nearest_geonames_pop():
    """(lat, lon, pop) triples for joining populations to curated cities."""
    path = Path(__file__).resolve().parent.parent / "data" / "raw" / "cities15000.txt"
    pts = []
    for line in path.read_text().splitlines():
        f = line.split("\t")
        if len(f) < 15 or f[6] != "P":
            continue
        try:
            pts.append((float(f[4]), float(f[5]), int(f[14])))
        except ValueError:
            continue
    return pts


def _norm(name):
    """Normalize a settlement name for matching."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    for junk in ["city of ", "municipality of "]:
        if s.startswith(junk):
            s = s[len(junk):]
    if s.endswith(" city"):
        s = s[: -len(" city")]
    return s


def _polygon_centroids():
    """Rough centroid per psgc10 from the migration repo's geojson:
    vertex mean of the largest ring (good enough to disambiguate)."""
    g = json.loads(
        (MIGRATION_REPO / "docs" / "municipal.geojson").read_text()
    )
    out = {}
    for f in g["features"]:
        geom = f["geometry"]
        polys = (
            geom["coordinates"]
            if geom["type"] == "MultiPolygon"
            else [geom["coordinates"]]
        )
        ring = max((p[0] for p in polys), key=len)
        lon = sum(pt[0] for pt in ring) / len(ring)
        lat = sum(pt[1] for pt in ring) / len(ring)
        out[str(f["properties"]["psgc10"])] = (lat, lon)
    return out


def ph_cities(top_n=80):
    """Top-N PH municipalities by 2024 population (+ PH_ALWAYS), with
    poblacion coordinates from OSM place nodes. Falls back to the
    municipality polygon centroid when no place node matches."""
    with open(MIGRATION_REPO / "data" / "clean" / "municipal_master.csv") as f:
        rows = [r for r in csv.DictReader(f) if r.get("pop2024")]
    for r in rows:
        r["name"] = r["name"].strip()
    rows.sort(key=lambda r: float(r["pop2024"]), reverse=True)
    chosen = rows[:top_n]
    have = {r["name"] for r in chosen}
    chosen += [r for r in rows[top_n:]
               if r["name"] in PH_ALWAYS and r["name"] not in have]

    centroids = _polygon_centroids()
    # The site geojson omits highly urbanized cities outside NCR (Cebu,
    # Baguio, Iloilo, ...). For those, disambiguate the OSM place node
    # against the province's center of mass instead.
    prov_pts, region_pts = {}, {}
    for r in rows:
        cen = centroids.get(str(r["psgc10"]))
        if cen:
            prov_pts.setdefault(r["province"], []).append(cen)
            region_pts.setdefault(r["region"], []).append(cen)

    def _centers(groups):
        return {
            k: (sum(c[0] for c in cs) / len(cs),
                sum(c[1] for c in cs) / len(cs))
            for k, cs in groups.items()
        }

    prov_center = _centers(prov_pts)
    region_center = _centers(region_pts)

    places = {}
    for el in fetch_ph_places().get("elements", []):
        nm = el.get("tags", {}).get("name")
        if nm:
            places.setdefault(_norm(nm), []).append(
                (el["lat"], el["lon"])
            )

    out = []
    for r in chosen:
        psgc = str(r["psgc10"])
        cen = centroids.get(psgc)
        # Reference point for disambiguation, tightest available first:
        # own polygon centroid (~55 km gate), province center (~2 deg),
        # region center (~3 deg). Highly urbanized cities sit outside
        # provinces (their province column holds the region), hence the
        # laddered fallback.
        ref, gate, how = None, None, ""
        for cand_ref, cand_gate, cand_how in [
            (cen, 0.25, ""),
            (prov_center.get(r["province"]), 4.0, "province_matched"),
            (region_center.get(r["region"]), 9.0, "region_matched"),
        ]:
            if cand_ref:
                ref, gate, how = cand_ref, cand_gate, cand_how
                break
        cands = places.get(_norm(r["name"]), [])
        latlon, flag = None, ""
        if cands and ref:
            best = min(
                cands,
                key=lambda c: (c[0] - ref[0]) ** 2 + (c[1] - ref[1]) ** 2,
            )
            if (best[0] - ref[0]) ** 2 + (best[1] - ref[1]) ** 2 < gate:
                latlon, flag = best, how
        if latlon is None and len(cands) == 1:
            latlon, flag = cands[0], "name_unique_match"
        if latlon is None and cen:
            latlon, flag = cen, "centroid_fallback"
        if latlon is None:
            continue
        slug = "ph-" + _norm(r["name"]).replace(" ", "-")
        if any(c["slug"] == slug for c in out):
            slug += "-" + _norm(r["province"]).replace(" ", "-")
        out.append(
            {
                "slug": slug,
                "name": r["name"],
                "country": "Philippines",
                "kind": "ph",
                "lat": round(latlon[0], 5),
                "lon": round(latlon[1], 5),
                "province": r["province"],
                "pop2024": float(r["pop2024"]),
                "pop": "",
                "region": "Asia",
                "netmig_rate": r.get("netmig_rate_per_1000_yr", ""),
                "income_class": r.get("income_class", ""),
                "flag": flag,
            }
        )
    return out


def all_cities(top_n_ph=150, top_n_world=None):
    info = _country_info()
    by_name = {v[0]: v[1] for v in info.values()}
    by_name.update({"Russia": by_name.get("Russian Federation", "Europe"),
                    "South Korea": "Asia", "Czechia": "Europe",
                    "Netherlands": "Europe"})
    geo = _nearest_geonames_pop()
    cities = []
    for slug, name, country, lat, lon in GLOBAL:
        near = [p for la, lo, p in geo
                if (la - lat) ** 2 + (lo - lon) ** 2 < 0.16]
        cities.append({
            "slug": slug, "name": name, "country": country,
            "kind": "global", "lat": lat, "lon": lon, "province": "",
            "pop2024": "", "pop": max(near) if near else "",
            "region": by_name.get(country, ""),
            "netmig_rate": "", "income_class": "", "flag": "",
        })
    return cities + world_top(top_n_world) + ph_cities(top_n_ph)

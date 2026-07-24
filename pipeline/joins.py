"""Joins to outside data: World Bank GDP per capita by country, and the
street-network sprawl index (SNDi) of Barrington-Leigh & Millard-Ball
(PNAS 2020) for the 188 cities they rank.

Raw files live in data/raw (wb_gdp_pcap.json, sndi_cities.html,
countryInfo.txt); nothing here fetches the network.
"""

import json
import re
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

COUNTRY_ISO3_ALIASES = {
    "Russia": "RUS", "South Korea": "KOR", "Czechia": "CZE",
    "Netherlands": "NLD",
}


def _country_table():
    """country name -> iso3, from GeoNames countryInfo."""
    out = {}
    for line in (RAW / "countryInfo.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        out[f[4]] = f[1]
    out.update(COUNTRY_ISO3_ALIASES)
    return out


def gdp_per_capita():
    """country name -> most recent GDP per capita (current US$)."""
    data = json.loads((RAW / "wb_gdp_pcap.json").read_text())[1]
    by_iso3 = {r["countryiso3code"]: r["value"]
               for r in data if r.get("value") is not None}
    return {name: round(by_iso3[iso3])
            for name, iso3 in _country_table().items()
            if iso3 in by_iso3}


def _norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def sndi_cities():
    """(normalized city name, normalized country) -> stock SNDi."""
    html = (RAW / "sndi_cities.html").read_text()
    out = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        cells = [re.sub(r"<[^>]+>", "", c).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
        if len(cells) < 6 or cells[0] == "City":
            continue
        try:
            out[(_norm(cells[0]), _norm(cells[1]))] = float(cells[5])
        except ValueError:
            continue
    return out


def join_sndi(cities):
    """Attach sndi to city dicts (name+country match, name-only backup)."""
    table = sndi_cities()
    by_name = {}
    for (nm, co), v in table.items():
        by_name.setdefault(nm, []).append((co, v))
    for c in cities:
        nm = _norm(c["name"].replace("City of ", "").split(" (")[0])
        co = _norm(c["country"])
        v = table.get((nm, co))
        if v is None:
            cands = by_name.get(nm, [])
            if len(cands) == 1:
                v = cands[0][1]
        c["sndi"] = v
    return cities

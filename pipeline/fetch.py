"""Overpass fetch with on-disk caching and a provenance log.

Every downloaded file is recorded in data/provenance.jsonl with the query,
endpoint, checksum, size, and retrieval time, following the convention of
the philippines-internal-migration repo.
"""

import hashlib
import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

_LOG_LOCK = threading.Lock()

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "overpass"
PROVENANCE = ROOT / "data" / "provenance.jsonl"

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
USER_AGENT = "street-rank/0.1 (research; github.com/DavidDemitriAfrica)"

# The street plan proper: drivable streets plus pedestrianised streets.
# Excludes service roads (parking aisles, driveways) and footpaths/trails,
# which would clutter modern cities with park paths.
HIGHWAY_RE = (
    "^(motorway|trunk|primary|secondary|tertiary|residential"
    "|unclassified|living_street|pedestrian)(_link)?$"
)


def _log_provenance(record):
    record["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    with _LOG_LOCK, open(PROVENANCE, "a") as f:
        f.write(json.dumps(record) + "\n")


def overpass(query, cache_name, max_tries=6):
    """Run an Overpass query, caching the JSON response under data/raw."""
    cache = RAW / cache_name
    if cache.exists() and cache.stat().st_size > 200:
        return json.loads(cache.read_text())

    last_err = None
    for attempt in range(max_tries):
        endpoint = ENDPOINTS[attempt % len(ENDPOINTS)]
        try:
            r = requests.post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=240,
            )
            if r.status_code == 200:
                data = r.json()
                if "elements" not in data:
                    raise ValueError("no elements key in response")
                cache.write_text(json.dumps(data))
                _log_provenance(
                    {
                        "file": str(cache.relative_to(ROOT)),
                        "endpoint": endpoint,
                        "query": query,
                        "bytes": cache.stat().st_size,
                        "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                        "osm_timestamp": data.get("osm3s", {}).get(
                            "timestamp_osm_base"
                        ),
                    }
                )
                return data
            last_err = f"HTTP {r.status_code}"
        except (requests.RequestException, ValueError, json.JSONDecodeError) as e:
            last_err = repr(e)
        time.sleep(8 * (attempt + 1))
    raise RuntimeError(f"overpass failed for {cache_name}: {last_err}")


def bbox_around(lat, lon, radius_m):
    """South, west, north, east of a box radius_m from (lat, lon)."""
    dlat = radius_m / 111320.0
    dlon = radius_m / (111320.0 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def fetch_streets(slug, lat, lon, radius_m):
    """All street ways with geometry inside the box around (lat, lon)."""
    s, w, n, e = bbox_around(lat, lon, radius_m)
    query = (
        f'[out:json][timeout:180];'
        f'way[highway~"{HIGHWAY_RE}"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});'
        f"out geom qt;"
    )
    return overpass(query, f"{slug}.json")


def fetch_ph_places():
    """Every place=city/town/municipality node in the Philippines."""
    query = (
        '[out:json][timeout:300];'
        'area["ISO3166-1"="PH"][admin_level=2]->.ph;'
        'node(area.ph)[place~"^(city|town|municipality)$"];'
        "out qt;"
    )
    return overpass(query, "_ph_places.json")

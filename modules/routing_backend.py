"""Distance-matrix backend: haversine (always available) or OSRM (real road-network
travel distance), selectable per solve.

Haversine remains the DEFAULT for the synthetic demo, and is the only sensible
choice there: the synthetic coordinates are drawn from a random radius around a
depot point and don't sit on a real road network, so a real routing engine would
return distances for roads that don't correspond to anything meaningful for made-up
points. OSRM is intended for the "upload your own dataset" path, where customers
have real-world coordinates and real road-network distance actually means something.

OSRM calls the public `router.project-osrm.org` demo server's `/table` endpoint
(no API key, rate-limited, not for production volume). On any network failure
(unreachable, timeout, non-200, malformed response) this falls back to haversine
and returns an explicit warning string — it never raises or silently returns wrong
distances.
"""
from __future__ import annotations

import functools

import numpy as np
import requests

EARTH_RADIUS_KM = 6371.0
OSRM_PUBLIC_ENDPOINT = "https://router.project-osrm.org"


def _haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    lat = np.radians(lats)[:, None]
    lon = np.radians(lons)[:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.clip(np.sqrt(a), -1, 1))


@functools.lru_cache(maxsize=32)
def _osrm_table_cached(coords_key: tuple, base_url: str, timeout: float) -> tuple:
    """coords_key: tuple of (lat, lon) pairs rounded to ~11cm precision, hashable so
    `lru_cache` can key on the exact location set — OSRM calls are comparatively
    expensive (network round-trip) so repeat solves over the same locations should
    not re-hit the network."""
    coord_str = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in coords_key)
    url = f"{base_url}/table/v1/driving/{coord_str}?annotations=distance"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM returned code={data.get('code')!r}")
    return tuple(tuple(row) for row in data["distances"])  # meters


def get_distance_matrix(lats: np.ndarray, lons: np.ndarray, mode: str = "haversine",
                         osrm_base_url: str = OSRM_PUBLIC_ENDPOINT, timeout: float = 5.0
                         ) -> tuple[np.ndarray, str, str | None]:
    """Returns (distance_km_matrix, mode_used, warning). `mode_used` differs from the
    requested `mode` only when OSRM was requested but unreachable — the caller (the
    Streamlit UI) should surface `warning` prominently rather than fail silently."""
    if mode not in ("haversine", "osrm"):
        raise ValueError(f"Unknown distance backend mode: {mode!r}")

    if mode == "haversine":
        return _haversine_matrix(lats, lons), "haversine", None

    try:
        coords_key = tuple(zip(np.round(lats, 6).tolist(), np.round(lons, 6).tolist()))
        distances_m = _osrm_table_cached(coords_key, osrm_base_url, timeout)
        return np.array(distances_m, dtype=float) / 1000.0, "osrm", None
    except Exception as exc:
        warning = (f"OSRM unreachable ({type(exc).__name__}: {exc}) — falling back to "
                   f"haversine distance for this solve.")
        return _haversine_matrix(lats, lons), "haversine", warning

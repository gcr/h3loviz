"""Sample a validated H3 lookup at display-projection pixel centers."""

import numpy as np
import pandas as pd
import shapely
import xarray as xr
import xdggs
from cartopy import crs as ccrs


def _bounds(bounds, fallback) -> tuple[float, float]:
    if bounds is None or any(value is None for value in bounds):
        bounds = fallback
    if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError("Viewport ranges must contain two finite, increasing bounds.")
    return tuple(map(float, bounds))


def sample_raster(
    values: pd.DataFrame,
    grid: xdggs.H3Info | None,
    projection: ccrs.Projection,
    nx: int,
    ny: int,
    x_range=None,
    y_range=None,
) -> xr.Dataset:
    xmin, xmax = _bounds(x_range, projection.x_limits)
    ymin, ymax = _bounds(y_range, projection.y_limits)
    x = xmin + (np.arange(nx) + 0.5) * (xmax - xmin) / nx
    y = ymin + (np.arange(ny) + 0.5) * (ymax - ymin) / ny
    samples = np.full((nx * ny, len(values.columns)), np.nan, dtype=np.float64)

    if grid is not None and not values.empty:
        xx, yy = np.meshgrid(x, y)
        px, py = xx.ravel(), yy.ravel()
        positions = np.flatnonzero(shapely.intersects_xy(projection.domain, px, py))
        lonlat = ccrs.PlateCarree().transform_points(
            projection, px[positions], py[positions]
        )
        lon, lat = lonlat[:, 0], lonlat[:, 1]
        valid = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) <= 90)
        if valid.any():
            ids = np.asarray(
                grid.geographic2cell_ids(lon[valid], lat[valid]), dtype=np.uint64
            )
            # One mapping and one lookup for all variables. Zeros remain zeros.
            samples[positions[valid], :] = values.reindex(ids).to_numpy()

    return xr.Dataset(
        {
            name: (("y", "x"), samples[:, i].reshape(ny, nx))
            for i, name in enumerate(values.columns)
        },
        coords={"y": y, "x": x},
        attrs={"bounds": (xmin, ymin, xmax, ymax)},
    )

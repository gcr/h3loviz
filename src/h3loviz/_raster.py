"""Sample H3 values, retaining only the latest viewport mapping and raster."""

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


class RasterSampler:
    """A bounded cache: coordinate/level changes reuse geometry where possible."""

    def __init__(self, projection, nx, ny):
        self.projection, self.nx, self.ny = projection, nx, ny
        self._mapping_key = None
        self._raster_key = None
        self._raster = None

    def sample(self, values, grid, revision, x_range=None, y_range=None):
        xmin, xmax = _bounds(x_range, self.projection.x_limits)
        ymin, ymax = _bounds(y_range, self.projection.y_limits)
        bounds = (xmin, ymin, xmax, ymax)
        key = (bounds, grid.level if grid is not None else None)
        raster_key = (key, revision)
        if raster_key == self._raster_key:
            return self._raster
        if key != self._mapping_key:
            x = xmin + (np.arange(self.nx) + 0.5) * (xmax - xmin) / self.nx
            y = ymin + (np.arange(self.ny) + 0.5) * (ymax - ymin) / self.ny
            positions = np.empty(0, dtype=int)
            ids = np.empty(0, dtype=np.uint64)
            if grid is not None:
                xx, yy = np.meshgrid(x, y)
                px, py = xx.ravel(), yy.ravel()
                positions = np.flatnonzero(
                    shapely.intersects_xy(self.projection.domain, px, py)
                )
                lonlat = ccrs.PlateCarree().transform_points(
                    self.projection, px[positions], py[positions]
                )
                lon, lat = lonlat[:, 0], lonlat[:, 1]
                valid = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) <= 90)
                positions = positions[valid]
                if valid.any():
                    ids = np.asarray(
                        grid.geographic2cell_ids(lon[valid], lat[valid]),
                        dtype=np.uint64,
                    )
            self._x, self._y, self._positions, self._ids = x, y, positions, ids
            self._mapping_key = key
        samples = np.full(
            (self.nx * self.ny, len(values.columns)), np.nan, dtype=np.float64
        )
        if len(self._ids) and not values.empty:
            samples[self._positions, :] = values.reindex(self._ids).to_numpy()
        self._raster = xr.Dataset(
            {
                name: (("y", "x"), samples[:, i].reshape(self.ny, self.nx))
                for i, name in enumerate(values.columns)
            },
            coords={"y": self._y, "x": self._x},
            attrs={"bounds": bounds},
        )
        self._raster_key = raster_key
        return self._raster


def sample_raster(
    values: pd.DataFrame,
    grid: xdggs.H3Info | None,
    projection: ccrs.Projection,
    nx: int,
    ny: int,
    x_range=None,
    y_range=None,
) -> xr.Dataset:
    return RasterSampler(projection, nx, ny).sample(values, grid, 0, x_range, y_range)

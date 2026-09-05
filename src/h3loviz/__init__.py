from typing import Any, Protocol, runtime_checkable

import geoviews as gv
import holoviews as hv
import numpy as np
import shapely
import xarray as xr
import xdggs
from cartopy import crs as ccrs
from holoviews.streams import RangeXY


class H3View:
    def __init__(
        self,
        df: Any,
        *,
        h3_level: int,
        nx: int = 320,
        ny: int = 200,
        h3_cell_column: str | None = None,
        projection: ccrs.Projection | None = None,
    ):
        self.geographic_proj = ccrs.PlateCarree()
        self.df = df
        self.h3_level = h3_level
        self.nx = nx
        self.ny = ny
        self.grid = xdggs.H3Info(level=self.h3_level)
        self.projection = projection or ccrs.PlateCarree()

        if h3_cell_column is not None:
            self.h3_cell_column = h3_cell_column
            if h3_cell_column not in self.df.columns:
                raise ValueError(f"Column '{h3_cell_column}' not found in DataFrame.")
        else:
            if "h3_cell" in self.df.columns:
                self.h3_cell_column = "h3_cell"
            elif "cell_id" in self.df.columns:
                self.h3_cell_column = "cell_id"
            elif "h3_cell_id" in self.df.columns:
                self.h3_cell_column = "h3_cell_id"
        # TODO: this should be easier
        self.df = self.df.set_index(self.h3_cell_column)

        # Updates when the user pans or zooms the map
        self.range_stream = RangeXY(
            x_range=self.geographic_proj.x_limits, y_range=self.geographic_proj.y_limits
        )

    def _sample_view(self, x_range, y_range):
        xmin, xmax = x_range or tuple(self.geographic_proj.x_limits)
        ymin, ymax = y_range or tuple(self.geographic_proj.y_limits)

        # Pixel centers, with the image edges exactly at the viewport bounds.
        x = xmin + (np.arange(self.nx) + 0.5) * (xmax - xmin) / self.nx
        y = ymin + (np.arange(self.ny) + 0.5) * (ymax - ymin) / self.ny

        xs = np.linspace(x_range[0], x_range[1], self.nx, dtype="float32")
        ys = np.linspace(y_range[0], y_range[1], self.ny, dtype="float32")

        xx, yy = np.meshgrid(xs, ys)
        px = xx.ravel()
        py = yy.ravel()

        # Equal Earth's rectangular bounding box includes space off the globe.
        inside = shapely.intersects_xy(self.geographic_proj.domain, px, py)
        print(inside,inside.shape)
        positions = np.flatnonzero(inside)
        print("positions",positions,positions.shape)
        print("px and py shape", px.shape, py.shape)
        lonlat = self.geographic_proj.transform_points(
            self.projection,
            px[positions],
            py[positions],
        )
        lon, lat = lonlat[:, 0], lonlat[:, 1]
        print("lon and lat shape", lon.shape, lat.shape)
        valid = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) <= 90)
        values = np.full(self.nx * self.ny, np.nan, dtype=np.float32)
        print("valid", valid, valid.shape)
        if valid.any():
            ids = np.asarray(
                self.grid.geographic2cell_ids(lon[valid], lat[valid]),
                dtype=np.uint64,
            )
            # Missing H3 cells remain NaN; real zero abundance stays zero.
            print("values",values,values.shape)
            print("positions[valid]",positions[valid],positions[valid].shape)
            values[positions[valid]] = self.df.reindex(ids).to_numpy().squeeze()
            # TODO: handle plotting other variables here

        raster = xr.DataArray(
            values.reshape(self.ny, self.nx),
            dims=("y", "x"),
            coords={"y": y, "x": x},
            name="abundance_weekly",
        )

        return gv.Image(
            raster,
            kdims=["x", "y"],
            vdims=["abundance_weekly"],
            crs=self.projection,
            bounds=(xmin, ymin, xmax, ymax),
        )


    def show(self):
        field = hv.DynamicMap(self._sample_view, streams=[self.range_stream])
        field = field.opts(
            projection=self.projection,
            cmap="plasma",
            # cnorm='log', clim=(0.001, 10.0),
            cnorm="eq_hist",
            colorbar=True,
            tools=["hover"],
            framewise=False,
            width=900,
            height=600,
        )
        self.range_stream.source = field
        return (field * gv.feature.coastline * gv.feature.borders)

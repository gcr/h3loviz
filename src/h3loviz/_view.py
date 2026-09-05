"""Public view configuration and HoloViews presentation."""

import warnings
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import geoviews as gv
import holoviews as hv
import xdggs
from cartopy import crs as ccrs
from holoviews.core.util import dimension_sanitizer
from holoviews.streams import RangeXY

from ._data import dimension_names, select_data
from ._raster import sample_raster


class H3View:
    """Render a coordinate slice of an eager pandas or Polars dataframe.

    ``vdims`` names the numeric columns sampled together; ``variable`` chooses
    the color field (default: the first vdim). ``coords`` declares columns to
    filter by scalar values in ``selection``. Singleton coordinates are selected
    automatically. Other columns are ignored.

    ``cell`` names the H3 column, or a named pandas index. When omitted, a single
    recognized column is used, falling back to a non-RangeIndex pandas index.
    Integer and hexadecimal-string IDs are validated after coordinate selection.
    A slice must have unique cells at a single inferred resolution. ``h3_level``
    is an optional assertion, not a resampling request.

    Input frames are not modified; the selected values are snapshotted on
    construction. Construct a new view to change data or selection in this
    release. Call ``hv.extension('bokeh')`` before displaying ``show()``.
    """

    def __init__(
        self,
        df: Any,
        *,
        vdims: Sequence[str],
        cell: str | None = None,
        coords: Sequence[str] = (),
        selection: Mapping[str, Any] | None = None,
        variable: str | None = None,
        h3_level: int | None = None,
        nx: int = 320,
        ny: int = 200,
        projection: ccrs.Projection | None = None,
    ):
        self.vdims = dimension_names(vdims, name="vdims")
        self.coords = dimension_names(coords, name="coords")
        if not self.vdims:
            raise ValueError("vdims must contain at least one numeric column.")
        if {"x", "y"} & set(self.vdims):
            raise ValueError(
                "Value names 'x' and 'y' are reserved for raster coordinates; rename these columns."
            )
        self.variable = self.vdims[0] if variable is None else variable
        if self.variable not in self.vdims:
            raise ValueError(f"variable={self.variable!r} must be included in vdims.")
        for name, value in (("nx", nx), ("ny", ny)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if h3_level is not None and (
            isinstance(h3_level, bool)
            or not isinstance(h3_level, Integral)
            or not 0 <= h3_level <= 15
        ):
            raise ValueError("h3_level must be an integer between 0 and 15.")
        selected = select_data(
            df,
            cell=cell,
            vdims=self.vdims,
            coords=self.coords,
            selection=selection,
            h3_level=h3_level,
        )
        self.df = selected.values
        self.cell = selected.cell
        self.h3_cell_column = selected.cell
        self.selection = selected.selection
        self.h3_level = selected.level
        self.grid = (
            xdggs.H3Info(level=self.h3_level) if self.h3_level is not None else None
        )
        self.nx, self.ny = int(nx), int(ny)
        self.projection = projection if projection is not None else ccrs.PlateCarree()
        self.range_stream = RangeXY(
            x_range=self.projection.x_limits, y_range=self.projection.y_limits
        )
        self._display = None

    def _sample_view(self, x_range=None, y_range=None):
        raster = sample_raster(
            self.df, self.grid, self.projection, self.nx, self.ny, x_range, y_range
        )
        ordered = [
            self.variable,
            *(name for name in self.vdims if name != self.variable),
        ]
        # Bokeh reserves these source fields for image data and its bounds.
        # Alias collisions only in the presentation layer, preserving labels.
        used = {"x", "y", "image", "dw", "dh"}
        source_names = {dimension_sanitizer(name) for name in ordered}
        renames = {}
        dimensions = []
        for i, name in enumerate(ordered):
            internal = name
            if dimension_sanitizer(internal) in used:
                internal = f"_h3loviz_value_{i}"
                while dimension_sanitizer(internal) in used | source_names:
                    internal += "_"
                renames[name] = internal
            used.add(dimension_sanitizer(internal))
            dimensions.append(hv.Dimension((internal, name)))
        return gv.Image(
            raster.rename(renames),
            kdims=["x", "y"],
            vdims=dimensions,
            crs=self.projection,
            bounds=raster.attrs["bounds"],
        )

    def show(self):
        """Return a reusable HoloViews overlay with viewport-driven sampling."""
        if self._display is None:
            field = hv.DynamicMap(self._sample_view, streams=[self.range_stream]).opts(
                projection=self.projection,
                cmap="plasma",
                cnorm="eq_hist",
                colorbar=True,
                tools=["hover"],
                framewise=False,
                width=900,
                height=600,
            )
            self.range_stream.source = field
            self._display = field * gv.feature.coastline * gv.feature.borders
        return self._display

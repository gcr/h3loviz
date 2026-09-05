"""A persistent H3 view with one atomic update path for Python and Panel."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import geoviews as gv
import holoviews as hv
import pandas as pd
import panel as pn
import xdggs
from cartopy import crs as ccrs
from holoviews.core.util import dimension_sanitizer
from holoviews.streams import RangeXY, Stream

from ._controls import make_controls
from ._data import (
    _resolve_cell,
    choose_selection,
    coordinate_table,
    dimension_names,
    select_data,
    snapshot_frame,
)
from ._raster import RasterSampler

_UNSET = object()
_ViewUpdate = Stream.define("H3ViewUpdate", revision=0)


@dataclass(frozen=True)
class _State:
    source: Any = None
    cell: str | None = None
    vdims: tuple[str, ...] = ()
    coords: tuple[str, ...] = ()
    select_coords: Mapping[str, Any] | None = None
    select_vdim: str | None = None
    level_assertion: int | None = None


class H3View(pn.viewable.Viewer):
    """A persistent viewport over an eager pandas or Polars dataframe.

    Declare numeric ``vdims`` and coordinate columns with ``coords``. Choose a
    coordinate combination with ``select_coords`` and its color field with
    ``select_vdim``. Unspecified selections retain compatible values, then fall
    back to an observed combination in coordinate declaration order.

    ``update`` patches named fields atomically. Coordinate dictionaries merge;
    ``select_coords=None`` resets them to automatic defaults. ``df=None`` clears
    the data while retaining configuration. A view stays blank until both data
    and vdims have been supplied. Loading and filtering files remain external.

    ``cell`` is inferred when unambiguous. Integer and hexadecimal-string cell
    IDs are validated after selection. Each slice must have unique cells at one
    resolution; ``h3_level`` optionally asserts that inferred resolution.

    ``selection`` and ``variable`` remain constructor aliases for the previous
    release. Use ``update`` rather than attribute assignment to change the view.
    """

    def __init__(
        self,
        df: Any = None,
        *,
        vdims: Sequence[str] | None = None,
        cell: str | None = None,
        coords: Sequence[str] = (),
        select_coords: Mapping[str, Any] | None = None,
        select_vdim: str | None = None,
        h3_level: int | None = None,
        nx: int = 320,
        ny: int = 200,
        projection: ccrs.Projection | None = None,
        selection=_UNSET,
        variable=_UNSET,
    ):
        super().__init__()
        if selection is not _UNSET:
            if select_coords is not None:
                raise ValueError("Use select_coords or its alias selection, not both.")
            select_coords = selection
        if variable is not _UNSET:
            if select_vdim is not None:
                raise ValueError("Use select_vdim or its alias variable, not both.")
            select_vdim = variable
        for name, value in (("nx", nx), ("ny", ny)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.nx, self.ny = int(nx), int(ny)
        self.projection = projection if projection is not None else ccrs.PlateCarree()
        self.range_stream = RangeXY(
            x_range=self.projection.x_limits, y_range=self.projection.y_limits
        )
        self.update_stream = _ViewUpdate()
        self._sampler = RasterSampler(self.projection, self.nx, self.ny)
        self._state = _State(select_coords={})
        self._table = pd.DataFrame()
        self._lookup = pd.DataFrame()
        self._data_revision = 0
        self._grid = None
        self._level = None
        self._resolved_cell = None
        self._display = None
        self._field = None
        self._controls_enabled = True
        self._controls = pn.Card(title="Selection", width=250, visible=False)
        self._error = pn.pane.Alert(alert_type="warning", visible=False)
        self._coordinate_widgets = {}
        self._vdim_widget = None
        self._widget_watchers = []
        self._syncing_widgets = False
        self.update(
            df=df,
            vdims=vdims,
            cell=cell,
            coords=coords,
            select_coords=select_coords,
            select_vdim=select_vdim,
            h3_level=h3_level,
        )

    @property
    def df(self):
        """The selected, normalized lookup; use update(df=...) to replace data."""
        return self._lookup

    @property
    def cell(self):
        return self._resolved_cell

    @property
    def vdims(self):
        return self._state.vdims

    @property
    def coords(self):
        return self._state.coords

    @property
    def select_coords(self):
        return dict(self._state.select_coords)

    @property
    def select_vdim(self):
        return self._state.select_vdim

    @property
    def selection(self):
        """Compatibility alias for select_coords."""
        return self.select_coords

    @property
    def variable(self):
        """Compatibility alias for select_vdim."""
        return self.select_vdim

    @property
    def h3_level(self):
        return self._level

    @property
    def grid(self):
        return self._grid

    def update(
        self,
        *,
        df=_UNSET,
        cell=_UNSET,
        vdims=_UNSET,
        coords=_UNSET,
        select_coords=_UNSET,
        select_vdim=_UNSET,
        h3_level=_UNSET,
    ) -> None:
        """Patch named fields, refresh controls, and notify the existing map once.

        Omitted arguments are unchanged. ``select_coords`` merges per key;
        passing None resets all coordinate selections. ``select_vdim=None``
        chooses the first vdim. Explicit choices must match observed data;
        incompatible retained choices fall back automatically. With no data,
        selections can be staged for subsequent loading. Invalid updates leave
        the source, selections, controls, and displayed raster unchanged.

        Returns None so a marimo update cell does not display a second map.
        """
        arguments = (df, cell, vdims, coords, select_coords, select_vdim, h3_level)
        if all(value is _UNSET for value in arguments):
            return
        old = self._state
        source = (
            old.source if df is _UNSET else (None if df is None else snapshot_frame(df))
        )
        values = (
            old.vdims
            if vdims is _UNSET
            else (() if vdims is None else dimension_names(vdims, name="vdims"))
        )
        if vdims is not _UNSET and vdims is not None and not values:
            raise ValueError(
                "vdims must contain at least one numeric column, or be None while configuring a view."
            )
        if {"x", "y"} & set(values):
            raise ValueError(
                "Value names 'x' and 'y' are reserved for raster coordinates; rename these columns."
            )
        dimensions = (
            old.coords if coords is _UNSET else dimension_names(coords, name="coords")
        )
        cell_spec = old.cell if cell is _UNSET else cell
        if cell_spec is not None and not isinstance(cell_spec, str):
            raise TypeError("cell must be a column/index name or None.")
        assertion = old.level_assertion if h3_level is _UNSET else h3_level
        if assertion is not None and (
            isinstance(assertion, bool)
            or not isinstance(assertion, Integral)
            or not 0 <= assertion <= 15
        ):
            raise ValueError("h3_level must be an integer between 0 and 15.")
        if set(values) & set(dimensions) or (
            cell_spec is not None and cell_spec in values + dimensions
        ):
            raise ValueError("Cell, value, and coordinate dimensions must be distinct.")
        if (
            select_coords is not _UNSET
            and select_coords is not None
            and not isinstance(select_coords, Mapping)
        ):
            raise TypeError(
                "select_coords must be a mapping of coordinate names to scalar values, or None."
            )
        explicit = (
            {}
            if select_coords is _UNSET or select_coords is None
            else dict(select_coords)
        )
        previous = {} if select_coords is None else old.select_coords
        metadata_changed = df is not _UNSET or coords is not _UNSET
        table = (
            coordinate_table(source, dimensions)
            if source is not None and metadata_changed
            else self._table
        )
        if source is None:
            table = pd.DataFrame()
        chosen = choose_selection(table, dimensions, previous, explicit)
        color = old.select_vdim if select_vdim is _UNSET else select_vdim
        if color is not None and not isinstance(color, str):
            raise TypeError("select_vdim must be a value column name or None.")
        if values and color not in values:
            if select_vdim is not _UNSET and select_vdim is not None:
                raise ValueError(f"select_vdim={color!r} must be included in vdims.")
            color = values[0]
        candidate = _State(
            source, cell_spec, values, dimensions, chosen, color, assertion
        )
        # Color-only changes reuse both the selected lookup and sampled raster.
        data_changed = any(
            value is not _UNSET
            for value in (df, cell, vdims, coords, select_coords, h3_level)
        )
        lookup, level, resolved_cell = self._lookup, self._level, self._resolved_cell
        if data_changed:
            if source is not None:
                resolved_cell, _ = _resolve_cell(
                    source, cell_spec, isinstance(source, pd.DataFrame)
                )
            else:
                resolved_cell = cell_spec
            if source is not None and values:
                selected = select_data(
                    source,
                    cell=cell_spec,
                    vdims=values,
                    coords=dimensions,
                    selection=chosen,
                    h3_level=assertion,
                )
                lookup, level, resolved_cell = (
                    selected.values,
                    selected.level,
                    selected.cell,
                )
            else:
                names = values or ("_h3loviz_empty",)
                lookup = pd.DataFrame(
                    {name: pd.Series(dtype="float64") for name in names}
                )
                level = None
        prepared = (
            make_controls(dimensions, values, chosen, color, table)
            if metadata_changed or vdims is not _UNSET
            else None
        )
        # Everything above is tentative. Commit only after successful validation.
        self._state = candidate
        self._table = table
        self._lookup, self._level, self._resolved_cell = lookup, level, resolved_cell
        self._grid = xdggs.H3Info(level=level) if level is not None else None
        if data_changed:
            self._data_revision += 1
        self._sync_controls(prepared=prepared)
        self._error.visible = False
        self.update_stream.event(revision=self.update_stream.revision + 1)

    def _sync_controls(self, *, prepared=None):
        self._syncing_widgets = True
        try:
            if prepared is not None:
                for widget, watcher in self._widget_watchers:
                    widget.param.unwatch(watcher)
                self._widget_watchers.clear()
                self._coordinate_widgets = {}
                self._vdim_widget = None
                for name, widget in prepared:
                    if name is None:
                        self._vdim_widget = widget
                    else:
                        self._coordinate_widgets[name] = widget

                    def changed(event, name=name):
                        self._widget_changed(name, event.new)

                    watcher = widget.param.watch(changed, "value")
                    self._widget_watchers.append((widget, watcher))
                self._controls.objects = [
                    *(widget for _, widget in prepared),
                    self._error,
                ]
            for name, widget in self._coordinate_widgets.items():
                widget.value = self.select_coords.get(name)
            if self._vdim_widget is not None:
                self._vdim_widget.value = self.select_vdim
            self._controls.visible = (
                self._controls_enabled
                and self._state.source is not None
                and bool(self.vdims)
            )
        finally:
            self._syncing_widgets = False

    def _widget_changed(self, name, value):
        if self._syncing_widgets:
            return
        try:
            if name is None:
                self.update(select_vdim=value)
            else:
                self.update(select_coords={name: value})
        except (ValueError, TypeError) as exc:
            self._sync_controls()
            self._error.object = str(exc)
            self._error.visible = True

    def _sample_view(self, x_range=None, y_range=None, revision=0):
        raster = self._sampler.sample(
            self.df, self.grid, self._data_revision, x_range, y_range
        )
        color = (
            self.select_vdim
            if self.select_vdim in raster.data_vars
            else next(iter(raster.data_vars))
        )
        ordered = [color, *(name for name in raster.data_vars if name != color)]
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
            label = "Value" if name == "_h3loviz_empty" and not self.vdims else name
            dimensions.append(hv.Dimension((internal, label)))
        return gv.Image(
            raster.rename(renames),
            kdims=["x", "y"],
            vdims=dimensions,
            crs=self.projection,
            bounds=raster.attrs["bounds"],
        )

    def show(self, *, controls: bool | None = None):
        """Return the same Panel layout on every call; optionally hide its controls."""
        if controls is not None:
            self._controls_enabled = controls
            self._sync_controls()
        if self._display is None:
            self._field = hv.DynamicMap(
                self._sample_view, streams=[self.range_stream, self.update_stream]
            ).opts(
                projection=self.projection,
                cmap="plasma",
                cnorm="eq_hist",
                colorbar=True,
                tools=["hover"],
                framewise=False,
                width=900,
                height=600,
            )
            self.range_stream.source = self._field
            self._pane = pn.pane.HoloViews(
                self._field * gv.feature.coastline * gv.feature.borders
            )
            self._display = pn.Row(self._pane, self._controls)
        return self._display

    def __panel__(self):
        return self.show()

# h3loviz

Render tabular H3 data as interactive raster maps in the HoloViz ecosystem.
Pass an eager pandas or Polars dataframe, select a coordinate slice, and pan or
zoom to resample its H3 values at the viewport's pixel centers.

This renderer looks up the cell containing each pixel center. It does **not**
aggregate observations with Datashader, draw every cell polygon, or compute
area-weighted pixel averages. At a coarse display scale, cells smaller than a
pixel can be missed. Aggregate observations into cells before passing them in.

## Install

```sh
uv add h3loviz
# For direct Polars input:
uv add 'h3loviz[polars]'
```

In this checkout, try the [marimo example](examples/coordinate_selection.py):

```sh
uv run --extra polars --with marimo marimo edit examples/coordinate_selection.py
```

## Display once, update from anywhere

```python
import holoviews as hv
from h3loviz import H3View

hv.extension("bokeh")

# Display cell: no data needed yet.
h3v = H3View(
    cell="h3_cell_id",
    vdims=["abundance_weekly", "checklists"],
    coords=["band", "h3_level"],
)
h3v.show()
```

In a separate reactive marimo cell, load or prepare your dataframe:

```python
# The notebook owns file selection, I/O, joins, and derived variables.
h3v.update(df=loaded_df)
```

Other cells can choose a slice or color independently:

```python
h3v.update(select_coords={"band": week_widget.value})
h3v.update(select_vdim="checklists")

# Change data and selections together in one update:
h3v.update(
    df=another_df,
    select_coords={"h3_level": 2},
    select_vdim="abundance_weekly",
)
```

The original map updates through a HoloViews stream. Automatic Panel controls
appear beside it once data and value dimensions are available. Numeric
coordinates use discrete sliders over observed values; categorical coordinates
use dropdowns. A color dropdown selects among `vdims`. Single-option controls
are disabled. Use `h3v.show(controls=False)` if marimo owns all controls.

The library uses HoloViews dimension metadata and Panel's widget machinery.
Panel controls and Python calls use the same `update` method, and Python updates
are reflected in the Panel controls. Marimo widgets send changes through cells;
they are not automatically synchronized back from Panel, nor does an update
rerun unrelated marimo cells. The example demonstrates both kinds of control.

## Update semantics

`update` changes only named arguments, validates the candidate state, and then
notifies the map once. Invalid updates raise without changing the source,
selection, controls, or raster. It returns `None`, avoiding an extra displayed
map when it is the last statement in a marimo cell.

Calls that patch different fields compose. When two cells write the same field,
the most recent call wins; replacing data may also adjust incompatible selections.

| Argument | Behavior |
| --- | --- |
| Omitted argument | Retain the current configuration or compatible selection. |
| `df=frame` | Snapshot an eager dataframe and refresh available coordinate values. |
| `df=None` | Clear the map while retaining configuration and selections. |
| `vdims=[...]` | Replace the numeric value columns sampled together. |
| `coords=[...]` | Replace the declared coordinate columns. |
| `select_coords={...}` | Merge selections by key; a week update retains the H3 level. |
| `select_coords={}` | Leave coordinate choices unchanged. |
| `select_coords=None` | Reset all coordinates to automatic defaults. |
| `select_vdim="..."` | Choose the color field from `vdims`; hover retains all values. |
| `select_vdim=None` | Choose the first vdim. |
| `cell="..."` / `cell=None` | Set the H3 column/index or request automatic detection. |
| `h3_level=n` / `h3_level=None` | Assert the selected resolution or clear that assertion. |

The data and `vdims` can arrive in either order. Until both are supplied, the
view stays blank. Declare `coords` before staging selections for them. Supply
schema changes in the same update as a new dataframe when its column names
change. No file loading takes place inside the library.

Explicit coordinate choices must match an observed combination. Missing choices
retain compatible previous values, then fall back in `coords` declaration order
to the first available values (sorted where comparable). Defaults always select
an existing combination, never a synthetic Cartesian product. For example,
changing species can automatically adjust its week if the previous week isn't
available. Choices explicitly supplied together must be compatible with each
other; a mismatch raises. An empty dataframe remains valid and renders blank;
its selections can be staged until data arrives.

Data updates preserve the viewport. Color changes reuse the current raster;
coordinate changes reuse the pixel-to-cell mapping when the H3 level is unchanged.
Pan, zoom, and H3-level changes compute a new mapping. Only the latest mapping
and raster are cached.

## Data interface

```python
H3View(
    df=None,
    *,
    vdims=None,             # Numeric columns; required before data can render
    cell=None,              # H3 column or named single-level pandas index
    coords=(),              # Explicit coordinate column names
    select_coords=None,     # Initial coordinate choices; defaults inferred
    select_vdim=None,       # Color field; defaults to the first vdim
    h3_level=None,          # Optional assertion against inferred resolution
    nx=320,
    ny=200,
    projection=None,        # Cartopy projection; defaults to PlateCarree
)
```

- **Cells:** integer and hexadecimal-string IDs are supported. IDs never pass
  through floating point during normalization. Null, invalid, and floating-point
  IDs are rejected. Without `cell=`, a single column named `h3_cell`, `cell_id`,
  `h3_cell_id`, `h3_idx`, or `h3_cell_id_str` is recognized. Multiple candidates
  require an explicit choice. With no recognized column, a non-RangeIndex,
  single-level pandas index is accepted and validated as H3 IDs. For arbitrary
  column names, use `cell="your_column"`.
- **Coordinates:** scalar equality selects rows; date strings remain strings.
  `None` selects null coordinates. Unrelated columns are ignored. Available
  combinations come from the loaded table, independently of H3 viewport sampling.
- **Resolution:** validation happens after selection. Every selected cell must
  have the same H3 resolution. `h3_level` checks it; it does not convert cells or
  select a column. Use `coords=["h3_level"]` and
  `select_coords={"h3_level": 2}` to choose a stored resolution.
- **Uniqueness:** each selected cell must have exactly one row. Duplicates require
  additional coordinate selection or upstream aggregation. Invalid slices reached
  through a Panel control restore the previous selection and display the error.
- **Missing data:** missing cells and null values become NaN/transparent pixels;
  real zeros stay zero. Empty input renders a blank field.
- **Inputs:** eager pandas and Polars dataframes only. Call `.df()`/`.pl()` on a
  DuckDB query or `.collect()` on a Polars lazy frame in the notebook first.
  The source is snapshotted for future selections, and selected values are copied
  into a pandas lookup. Source frames are never modified. Value arrays use
  float64. `x` and `y` are reserved raster coordinate names.

The sampled viewport is an xarray `Dataset` with one `(y, x)` array per value
column. The full source remains tabular; no cells × weeks × species cube is
constructed. `h3v.df` exposes the selected lookup for inspection.

`show()` returns a persistent Panel layout containing the HoloViews map and
controls. `pn.panel(h3v)` and rich notebook display use that same layout.
Display requires the Bokeh extension and a live Python notebook/server. Cartopy
may download its boundary datasets on first display. Runtime updates are made
through `update`, rather than by mutating attributes or `h3v.df`.

The previous constructor keywords `selection` and `variable` are retained as
aliases for `select_coords` and `select_vdim`; their read-only properties also
remain available. Automatic coordinate defaults replace the earlier requirement
to specify every non-singleton coordinate. The return value of `show()` is now
a Panel layout rather than a raw HoloViews overlay.

## Development

```sh
uv run --extra polars --with pytest pytest
uv run --with ruff ruff check src tests examples
uv run --with marimo marimo check examples/coordinate_selection.py
uv build
```

Tests cover both dataframe backends, H3 validation, projections, missing values,
live Bokeh updates, Panel controls, atomic failure, source replacement, viewport
preservation, and cache reuse. They use synthetic data and do not download
boundary datasets.

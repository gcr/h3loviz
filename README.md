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

In this checkout, run the example below with `uv run --extra polars --with marimo marimo edit
examples/coordinate_selection.py` (or use its inline dependencies with
`uvx marimo edit --sandbox examples/coordinate_selection.py`).

## A small example

```python
import h3
import holoviews as hv
import pandas as pd
from h3loviz import H3View

hv.extension("bokeh")

cells = sorted(h3.grid_disk(h3.latlng_to_cell(42.44, -76.50, 4), 2))
df = pd.DataFrame([
    {
        "h3_cell_id": cell,
        "week": week,
        "abundance": float(i + week),
        "checklists": (i + 1) * 10,
    }
    for week in [1, 2]
    for i, cell in enumerate(cells)
])

h3v = H3View(
    df,
    vdims=["abundance", "checklists"],
    coords=["week"],
    selection={"week": 2},
    variable="abundance",
)
h3v.show()  # Last expression in a notebook cell; zoom to the Finger Lakes.
```

Both variables are sampled together. `variable` chooses the color field; hover
shows all supplied value dimensions. The default color field is `vdims[0]`.
The sampled viewport is an xarray `Dataset` with one `(y, x)` array per variable.
The source is kept tabular during filtering: no cells × weeks × species cube
is constructed.

## Data interface

```python
H3View(
    df,
    *,
    vdims,                  # Required sequence of real numeric column names
    cell=None,              # H3 column or named single-level pandas index
    coords=(),              # Coordinate column names, declared explicitly
    selection=None,         # Mapping of coordinate names to scalar values
    variable=None,          # Color field; defaults to the first vdim
    h3_level=None,           # Optional assertion against inferred resolution
    nx=320,
    ny=200,
    projection=None,        # Cartopy projection; defaults to PlateCarree
    h3_cell_column=None,    # Deprecated alias for cell
)
```

- **Cells:** integer and hexadecimal-string IDs are supported. IDs never pass
  through floating point during normalization. Null, invalid, and floating-point
  IDs are rejected. Without `cell=`, a single column named `h3_cell`, `cell_id`,
  `h3_cell_id`, `h3_idx`, or `h3_cell_id_str` is recognized. Multiple candidates
  require an explicit choice. With no recognized column, a non-RangeIndex,
  single-level pandas index is accepted and validated as H3 IDs. For arbitrary
  column names, use `cell="your_column"`.
- **Coordinates:** every declared coordinate with multiple values in the input
  needs an explicit scalar selection. Singleton coordinates are selected
  automatically. Selection uses exact equality (including date strings as
  strings); `None` selects null coordinates. Unrelated columns are ignored.
- **Resolution:** validation happens after selection. Every selected cell must
  have the same H3 resolution. `h3_level` checks this value; it does not convert
  cells or select a column. Use `coords=["h3_level"]` and
  `selection={"h3_level": 2}` to choose a stored resolution.
- **Uniqueness:** each selected cell must have exactly one row. Duplicate cells
  produce an error asking for additional filtering or upstream aggregation.
- **Missing data:** missing cells and null values become NaN/transparent pixels;
  real zeros stay zero. An empty input or selection produces a blank field.
  Without cells or an explicit assertion, its `h3_level` is `None`.
- **Inputs:** eager pandas and Polars dataframes only. Call `.df()`/`.pl()` on a
  DuckDB query or `.collect()` on a Polars lazy frame in the notebook first.
  Selected values are copied into a pandas lookup at construction; source frames
  are never modified. Value arrays use float64. `x` and `y` are reserved raster
  coordinate names and cannot be value dimensions.

`show()` returns a reusable HoloViews overlay with coastlines, borders, and a
`RangeXY` stream. Pan/zoom performs one H3 mapping and one lookup for all value
columns without re-ingesting the source. Display requires the Bokeh extension
and a live Python notebook/server. Cartopy may download its boundary datasets
on first display.

In this release, construct a new `H3View` to change data, coordinates, or the
color variable. Mutating view attributes or `h3v.df` is not an update API.
The old `h3_level=` argument still works as an assertion and `h3_cell_column=`
emits a deprecation warning; callers must now supply `vdims` explicitly.

## Notebook-owned loading

For the eBird-style parquet schema, the notebook can do the expensive loading
and filtering itself:

```python
# In a reactive marimo cell depending on the file/week/level widgets:
df = pl.read_parquet(file_widget.value).filter(
    pl.col("band") == week_widget.value,
    pl.col("h3_level") == level_widget.value,
)
H3View(df, cell="h3_cell_id", vdims=["abundance_weekly"]).show()
```

`cell=` is explicit here because the parquet contains both integer and string
H3 columns. Keep file selection, I/O, joins, and derived variables in the
notebook. See [the synthetic marimo example](examples/coordinate_selection.py)
for working coordinate and color controls using the current constructor API.

### Planned follow-up — not implemented yet

A later milestone will support an initially empty, persistent display and a
stream-driven update path:

```python
# Future API, display cell:
h3v = H3View(df=None)
h3v.show()

# Future API, notebook-owned loading cell:
h3v.set_df(
    loaded_and_filtered_df,
    cell="h3_cell_id",
    vdims=["abundance_weekly"],
    coords=["band", "h3_level"],
)
```

Panel controls and programmatic selection will use the same update path;
marimo cells can feed new data through it. Automatic coordinate widgets,
persistent data replacement, and rich display hooks belong to that milestone.
File loading remains outside the library.

## Development

```sh
uv run --extra polars --with pytest pytest
uv run --with ruff ruff check src tests examples
uv build
```

Tests use synthetic H3 cells and exercise both input backends, coordinate and
resolution validation, projection sampling, hover fields, and stream updates.
They do not require the bird datasets or download boundary shapes.

import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import shapely
from cartopy import crs as ccrs
from h3.api import basic_int as h3

from h3loviz import H3View
from h3loviz._raster import sample_raster


@pytest.fixture(scope="module", autouse=True)
def bokeh():
    hv.extension("bokeh")


def test_pixel_centers_variables_and_missing_cells():
    # Four distinct cells at the centers, deliberately far from viewport edges.
    ids = [h3.latlng_to_cell(lat, lon, 4) for lat in [-10, 10] for lon in [-10, 10]]
    df = pd.DataFrame(
        {"h3_cell_id": ids[:3], "abundance": [0.0, 2.0, 3.0], "count": [10, 20, 30]}
    )
    view = H3View(df, vdims=["abundance", "count"], variable="count", nx=2, ny=2)
    image = view._sample_view((-20, 20), (-20, 20))
    assert image.bounds.lbrt() == (-20, -20, 20, 20)
    assert [dim.name for dim in image.vdims] == ["count", "abundance"]
    np.testing.assert_equal(image.data.x.values, [-10.0, 10.0])
    np.testing.assert_equal(image.data.y.values, [-10.0, 10.0])
    np.testing.assert_equal(image.data.abundance.values, [[0.0, 2.0], [3.0, np.nan]])
    np.testing.assert_equal(image.data["count"].values, [[10.0, 20.0], [30.0, np.nan]])


@pytest.mark.parametrize("projection", [ccrs.PlateCarree(), ccrs.EqualEarth()])
def test_inverse_projection_at_known_location(projection):
    cell = h3.latlng_to_cell(42, -76, 4)
    lon, lat = -76.0, 42.0
    px, py = projection.transform_point(lon, lat, ccrs.PlateCarree())
    df = pd.DataFrame({"h3_cell_id": [cell], "value": [7.0]})
    view = H3View(df, vdims=["value"], projection=projection, nx=1, ny=1)
    image = view._sample_view((px - 0.01, px + 0.01), (py - 0.01, py + 0.01))
    assert image.data.value.item() == 7.0
    assert view.range_stream.x_range == projection.x_limits
    assert view.range_stream.y_range == projection.y_limits


def test_equal_earth_off_globe_mask_and_default_ranges():
    projection = ccrs.EqualEarth()
    df = pd.DataFrame({"h3_cell_id": list(h3.get_res0_cells()), "value": 1.0})
    view = H3View(df, vdims=["value"], projection=projection, nx=30, ny=20)
    image = view._sample_view(None, (None, None))
    raster = image.data
    xx, yy = np.meshgrid(raster.x, raster.y)
    inside = shapely.intersects_xy(projection.domain, xx, yy)
    assert inside.any() and (~inside).any()
    assert np.isnan(raster.value.values[~inside]).all()
    np.testing.assert_equal(raster.value.values[inside], 1.0)
    assert image.bounds.lbrt() == (
        *projection.x_limits[:1],
        *projection.y_limits[:1],
        projection.x_limits[1],
        projection.y_limits[1],
    )


def test_one_mapping_per_viewport_and_no_reingestion(monkeypatch):
    cell = h3.latlng_to_cell(0, 0, 4)
    view = H3View(
        pd.DataFrame({"h3_cell_id": [cell], "a": [1.0], "b": [2.0]}), vdims=["a", "b"]
    )
    calls = []
    original = type(view.grid).geographic2cell_ids

    def tracked(grid, lon, lat):
        calls.append(len(lon))
        return original(grid, lon, lat)

    def unexpected(*args, **kwargs):
        raise AssertionError("Pan/zoom must not normalize the source again")

    monkeypatch.setattr(type(view.grid), "geographic2cell_ids", tracked)
    monkeypatch.setattr("h3loviz._view.select_data", unexpected)
    display = view.show()
    assert view.show() is display
    field = view.range_stream.source
    field[()]
    view.range_stream.event(x_range=(-10, 10), y_range=(-10, 10))
    field[()]
    assert len(calls) == 2


def test_bokeh_color_and_hover():
    df = pd.DataFrame(
        {"h3_cell_id": list(h3.get_res0_cells()), "abundance": 1.0, "count": 2.0}
    )
    view = H3View(df, vdims=["abundance", "count"], variable="count", nx=4, ny=4)
    image = view._sample_view().opts(tools=["hover"])
    plot = hv.renderer("bokeh").get_plot(image)
    tooltips = dict(plot.handles["hover"].tooltips)
    assert tooltips["count"] == "@image"
    assert tooltips["abundance"] == "@{abundance}"
    np.testing.assert_equal(plot.handles["source"].data["image"][0], 2.0)
    np.testing.assert_equal(plot.handles["source"].data["abundance"][0], 1.0)


def test_empty_input_and_invalid_bounds():
    df = pd.DataFrame(
        {"h3_cell_id": pd.Series(dtype="uint64"), "value": pd.Series(dtype="float64")}
    )
    view = H3View(df, vdims=["value"])
    assert np.isnan(view._sample_view().data.value).all()
    with pytest.raises(ValueError, match="increasing bounds"):
        sample_raster(view.df, view.grid, view.projection, 2, 2, (1, 1), (-1, 1))


@pytest.mark.parametrize(
    "names",
    [
        ("image", "safe"),
        ("safe", "image"),
        ("safe", "dw"),
        ("safe", "dh"),
        ("a b", "a_b"),
    ],
)
def test_value_names_cannot_overwrite_bokeh_image_fields(names):
    primary, secondary = names
    df = pd.DataFrame(
        {"cell_id": list(h3.get_res0_cells()), primary: 1.0, secondary: 2.0}
    )
    view = H3View(df, vdims=names, nx=2, ny=2)
    image = view._sample_view().opts(tools=["hover"])
    plot = hv.renderer("bokeh").get_plot(image)
    source = plot.handles["source"].data
    tooltips = dict(plot.handles["hover"].tooltips)
    assert tooltips[primary] == "@image"
    np.testing.assert_equal(source["image"][0], 1.0)
    secondary_field = tooltips[secondary].removeprefix("@{").removesuffix("}")
    np.testing.assert_equal(source[secondary_field][0], 2.0)

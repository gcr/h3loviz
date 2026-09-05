from datetime import date

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import polars as pl
import pytest
from h3.api import basic_int as h3

from h3loviz import H3View


@pytest.fixture(scope="module", autouse=True)
def bokeh():
    hv.extension("bokeh")


@pytest.fixture(params=[pd.DataFrame, pl.DataFrame], ids=["pandas", "polars"])
def source(request):
    rows = [
        {"cell_id": cell, "week": week, "a": float(week), "b": 10.0 + week}
        for week in [1, 4]
        for cell in h3.get_res0_cells()
    ]
    return request.param(rows)


def test_blank_then_incremental_configuration(source):
    view = H3View(nx=4, ny=4)
    layout = view.show()
    plot = hv.renderer("bokeh").get_plot(view.range_stream.source)
    model = plot.state
    assert np.isnan(plot.handles["source"].data["image"][0]).all()
    assert not view._controls.visible
    assert view.update(df=source) is None
    assert np.isnan(plot.handles["source"].data["image"][0]).all()
    view.update(coords=["week"], select_coords={"week": 4})
    view.update(vdims=["a", "b"], select_vdim="b")
    assert view.show() is layout
    assert plot.state is model
    assert view._controls.visible
    assert view.select_coords == {"week": 4}
    np.testing.assert_equal(plot.handles["source"].data["image"][0], 14.0)
    assert dict(plot.handles["hover"].tooltips)["a"] == "@{a}"


def test_controls_and_python_share_state(source):
    view = H3View(source, vdims=["a", "b"], coords=["week"], nx=4, ny=4)
    view.show()
    plot = hv.renderer("bokeh").get_plot(view.range_stream.source)
    widget = view._coordinate_widgets["week"]
    assert isinstance(widget, pn.widgets.DiscreteSlider)
    assert list(widget.options.values()) == [1, 4]
    start = view.update_stream.revision
    widget.value = 4
    assert view.select_coords == {"week": 4}
    assert view.update_stream.revision == start + 1
    np.testing.assert_equal(plot.handles["source"].data["image"][0], 4.0)
    view._vdim_widget.value = "b"
    assert view.select_vdim == "b"
    np.testing.assert_equal(plot.handles["source"].data["image"][0], 14.0)
    view.update(select_coords={"week": 1}, select_vdim="a")
    assert widget.value == 1
    assert view._vdim_widget.value == "a"
    np.testing.assert_equal(plot.handles["source"].data["image"][0], 1.0)
    assert view._coordinate_widgets["week"] is widget


def test_coordinate_updates_merge_and_reset():
    df = pd.DataFrame(
        [
            {
                "cell_id": h3.latlng_to_cell(0, 0, level),
                "level": level,
                "week": week,
                "a": 1.0,
            }
            for level in [2, 3]
            for week in [1, 4]
        ]
    )
    view = H3View(df, vdims=["a"], coords=["week", "level"], select_coords={"level": 3})
    view.update(select_coords={"week": 4})
    assert view.select_coords == {"week": 4, "level": 3}
    assert view.h3_level == 3
    view.update(select_coords=None)
    assert view.select_coords == {"week": 1, "level": 2}
    assert view.h3_level == 2
    view.update(select_coords={"week": 4})
    view.update(select_coords={})
    assert view.select_coords == {"week": 4, "level": 2}


def test_defaults_use_observed_combinations_and_adapt_other_controls():
    cell = h3.latlng_to_cell(0, 0, 2)
    df = pd.DataFrame(
        {
            "cell_id": [cell, cell],
            "species": ["crow", "raven"],
            "week": [4, 1],
            "a": [2.0, 3.0],
        }
    )
    view = H3View(df, vdims=["a"], coords=["species", "week"])
    assert view.select_coords == {"species": "crow", "week": 4}
    view._coordinate_widgets["species"].value = "raven"
    assert view.select_coords == {"species": "raven", "week": 1}
    assert view._coordinate_widgets["week"].value == 1
    view.update(select_coords={"week": 4})
    assert view.select_coords == {"species": "crow", "week": 4}
    with pytest.raises(ValueError, match="No rows match"):
        view.update(select_coords={"species": "crow", "week": 1})
    assert view.select_coords == {"species": "crow", "week": 4}


def test_replacement_preserves_selection_then_falls_back_and_detaches_widgets(source):
    view = H3View(
        source,
        vdims=["a", "b"],
        coords=["week"],
        select_coords={"week": 4},
        select_vdim="b",
    )
    old_widget = view._coordinate_widgets["week"]
    view.update(df=source)
    assert view.select_coords == {"week": 4}
    assert view.select_vdim == "b"
    revision = view.update_stream.revision
    old_widget.value = 1
    assert view.update_stream.revision == revision
    replacement = pd.DataFrame(
        {"cell_id": [h3.latlng_to_cell(0, 0, 2)], "week": [9], "c": [5.0]}
    )
    view.update(df=replacement, vdims=["c"])
    assert view.select_coords == {"week": 9}
    assert view.select_vdim == "c"
    assert view._coordinate_widgets["week"].disabled
    assert view._vdim_widget.disabled


def test_invalid_update_is_atomic(source):
    view = H3View(source, vdims=["a", "b"], coords=["week"])
    view.show()
    plot = hv.renderer("bokeh").get_plot(view.range_stream.source)
    before = view._state
    lookup = view.df
    revision = view.update_stream.revision
    widget = view._coordinate_widgets["week"]
    raster = plot.handles["source"].data["image"][0].copy()
    bad = pd.DataFrame({"cell_id": [0], "week": [7], "a": [6.0], "b": [8.0]})
    for changes in (
        {"df": bad, "select_coords": {"week": 7}},
        {"select_coords": {"week": 99}, "select_vdim": "b"},
        {"select_vdim": "missing"},
        {"vdims": ["missing"]},
    ):
        with pytest.raises((ValueError, TypeError)):
            view.update(**changes)
        assert view._state is before
        assert view.df is lookup
        assert view.update_stream.revision == revision
        assert view._coordinate_widgets["week"] is widget
        np.testing.assert_equal(plot.handles["source"].data["image"][0], raster)


def test_widget_failure_restores_selection_and_explains_error():
    cell = h3.latlng_to_cell(0, 0, 2)
    df = pd.DataFrame({"cell_id": [cell] * 3, "week": [1, 4, 4], "a": [1.0, 2.0, 3.0]})
    view = H3View(df, vdims=["a"], coords=["week"])
    view._coordinate_widgets["week"].value = 4
    assert view.select_coords == {"week": 1}
    assert view._coordinate_widgets["week"].value == 1
    assert view._error.visible
    assert "duplicate H3" in view._error.object


def test_clear_and_repopulate_reuses_layout_and_viewport(source):
    view = H3View(source, vdims=["a", "b"], coords=["week"], nx=4, ny=4)
    layout = view.show(controls=False)
    plot = hv.renderer("bokeh").get_plot(view.range_stream.source)
    plot.state.x_range.start, plot.state.x_range.end = -20, 20
    plot.state.y_range.start, plot.state.y_range.end = -10, 10
    view.range_stream.event(x_range=(-20, 20), y_range=(-10, 10))
    view.update(select_coords={"week": 4})
    view.update(df=None)
    assert np.isnan(plot.handles["source"].data["image"][0]).all()
    assert view.select_coords == {"week": 4}
    assert not view._controls.visible
    view.update(df=source)
    assert view.show() is layout
    assert not view._controls.visible
    assert view.range_stream.x_range == (-20, 20)
    assert (plot.state.x_range.start, plot.state.x_range.end) == (-20, 20)
    np.testing.assert_equal(plot.handles["source"].data["image"][0], 4.0)
    view.show(controls=True)
    assert view._controls.visible
    assert view.__panel__() is layout


def test_cache_reuses_mapping_and_color_raster(source, monkeypatch):
    view = H3View(source, vdims=["a", "b"], coords=["week"], nx=4, ny=4)
    view._sample_view()
    mapping = view._sampler._ids
    raster = view._sampler._raster

    def unexpected(*args, **kwargs):
        raise AssertionError("Unnecessary ingestion or pixel mapping")

    with monkeypatch.context() as context:
        context.setattr("h3loviz._view.select_data", unexpected)
        context.setattr(type(view.grid), "geographic2cell_ids", unexpected)
        view.update(select_vdim="b")
        view._sample_view()
    assert view._sampler._raster is raster
    with monkeypatch.context() as context:
        context.setattr(type(view.grid), "geographic2cell_ids", unexpected)
        view.update(select_coords={"week": 4})
        view._sample_view()
    assert view._sampler._ids is mapping
    assert view._sampler._raster is not raster
    view._sample_view((-10, 10), (-10, 10))
    assert view._sampler._ids is not mapping


def test_level_change_invalidates_mapping():
    df = pd.DataFrame(
        {
            "cell_id": [h3.latlng_to_cell(0, 0, r) for r in [2, 3]],
            "level": [2, 3],
            "a": [1.0, 2.0],
        }
    )
    view = H3View(df, vdims=["a"], coords=["level"], nx=4, ny=4)
    view._sample_view()
    mapping = view._sampler._ids
    view.update(select_coords={"level": 3})
    view._sample_view()
    assert view._sampler._ids is not mapping
    assert view.h3_level == 3


def test_future_selection_uses_source_snapshot():
    cell = h3.latlng_to_cell(0, 0, 2)
    df = pd.DataFrame({"cell_id": [cell, cell], "week": [1, 4], "a": [1.0, 4.0]})
    view = H3View(df, vdims=["a"], coords=["week"])
    df.loc[1, "a"] = 1000.0
    view.update(select_coords={"week": 4})
    assert view.df.loc[cell, "a"] == 4.0


@pytest.mark.parametrize(
    "values", [[None, "raven"], [date(2026, 1, 1), date(2026, 2, 1)]]
)
def test_null_and_date_coordinate_controls(values):
    cell = h3.latlng_to_cell(0, 0, 2)
    df = pd.DataFrame({"cell_id": [cell, cell], "coord": values, "a": [1.0, 2.0]})
    view = H3View(df, vdims=["a"], coords=["coord"])
    view.update(select_coords={"coord": values[0]})
    assert view._coordinate_widgets["coord"].value == values[0]
    view._coordinate_widgets["coord"].value = values[1]
    assert view.select_coords["coord"] == values[1]


def test_staged_selection_and_empty_frame():
    view = H3View(coords=["week"])
    view.update(select_coords={"week": 4}, select_vdim="a")
    assert view.select_coords == {"week": 4}
    view.update(vdims=["a"])
    view.update(
        df=pd.DataFrame(
            {
                "cell_id": pd.Series(dtype="uint64"),
                "week": pd.Series(dtype="int64"),
                "a": pd.Series(dtype="float64"),
            }
        )
    )
    assert view.df.empty
    assert view._coordinate_widgets["week"].disabled
    assert view.select_coords == {"week": 4}


def test_selection_aliases_and_noop(source):
    view = H3View(
        source, vdims=["a", "b"], coords=["week"], selection={"week": 4}, variable="b"
    )
    assert view.select_coords == view.selection == {"week": 4}
    assert view.select_vdim == view.variable == "b"
    revision = view.update_stream.revision
    view.update()
    assert view.update_stream.revision == revision
    with pytest.raises(ValueError, match="not both"):
        H3View(select_coords={"week": 1}, selection={"week": 4})
    with pytest.raises(ValueError, match="not both"):
        H3View(select_vdim="a", variable="b")


def test_schema_replacement_updates_existing_glyph_and_hover(source):
    view = H3View(source, vdims=["a", "b"], coords=["week"], nx=4, ny=4)
    view.show()
    plot = hv.renderer("bokeh").get_plot(view.range_stream.source)
    model = plot.state
    replacement = pd.DataFrame({"cell_id": list(h3.get_res0_cells()), "new_value": 7.0})
    view.update(df=replacement, coords=[], vdims=["new_value"])
    assert plot.state is model
    assert view.select_coords == {}
    assert view.select_vdim == "new_value"
    tooltips = dict(plot.handles["hover"].tooltips)
    assert tooltips["new_value"] == "@image"
    assert "a" not in tooltips and "b" not in tooltips
    np.testing.assert_equal(plot.handles["source"].data["image"][0], 7.0)


@pytest.mark.parametrize("changes", [{"select_vdim": []}, {"cell": []}])
def test_invalid_staged_configuration_is_rejected(changes):
    view = H3View()
    state = view._state
    with pytest.raises(TypeError):
        view.update(**changes)
    assert view._state is state

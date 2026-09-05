import numpy as np
import pandas as pd
import polars as pl
import pytest
from h3.api import basic_int as h3

from h3loviz import H3View

CELL = h3.latlng_to_cell(42, -76, 4)
OTHER = h3.latlng_to_cell(30, -90, 4)


@pytest.fixture(params=[pd.DataFrame, pl.DataFrame], ids=["pandas", "polars"])
def frame(request):
    return request.param


@pytest.mark.parametrize("string_ids", [False, True])
def test_backends_select_snapshot_and_normalize(frame, string_ids):
    ids = [CELL, CELL, OTHER]
    if string_ids:
        ids = [format(cell, "x") for cell in ids]
    df = frame(
        {
            "h3_cell_id": ids,
            "week": [1, 2, 2],
            "species": ["raven"] * 3,
            "abundance": [99.0, 0.0, 2.0],
            "count": [9, 4, 7],
            "ignored": ["a", "b", "c"],
        }
    )
    original = df.copy(deep=True) if isinstance(df, pd.DataFrame) else df.clone()
    view = H3View(
        df,
        vdims=["abundance", "count"],
        coords=["week", "species"],
        selection={"week": 2},
    )
    assert view.h3_level == 4
    assert view.selection == {"week": 2, "species": "raven"}
    assert view.df.index.dtype == np.dtype("uint64")
    assert list(view.df.index) == [CELL, OTHER]
    np.testing.assert_equal(view.df.to_numpy(), [[0.0, 4.0], [2.0, 7.0]])
    if isinstance(df, pd.DataFrame):
        pd.testing.assert_frame_equal(df, original)
        df.loc[1, "abundance"] = 1000
        assert view.df.loc[CELL, "abundance"] == 0
    else:
        assert df.equals(original)


@pytest.mark.parametrize("name", [None, "my_cells"])
@pytest.mark.parametrize("string_ids", [False, True])
def test_pandas_index(name, string_ids):
    ids = [format(CELL, "x")] if string_ids else [CELL]
    df = pd.DataFrame({"value": [1.0]}, index=pd.Index(ids, name=name))
    view = H3View(df, vdims=["value"])
    assert list(view.df.index) == [CELL]
    if name:
        pd.testing.assert_frame_equal(
            view.df, H3View(df, cell=name, vdims=["value"]).df
        )


def test_ambiguous_columns_require_override(frame):
    df = frame(
        {"h3_cell_id": [CELL], "h3_cell_id_str": [format(CELL, "x")], "value": [1.0]}
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        H3View(df, vdims=["value"])
    assert H3View(df, cell="h3_cell_id", vdims=["value"]).h3_level == 4


@pytest.mark.parametrize(
    "bad", [None, 0, -1, 2**64, float(CELL), True, "nope", "fffffffffffffff"]
)
def test_invalid_ids(bad):
    df = pd.DataFrame({"h3_cell_id": pd.Series([bad], dtype=object), "value": [1.0]})
    with pytest.raises(ValueError, match="Invalid H3"):
        H3View(df, vdims=["value"])


def test_nullable_h3_ids_rejected_without_float_conversion(frame):
    if frame is pd.DataFrame:
        df = frame(
            {"h3_cell_id": pd.array([CELL, None], dtype="UInt64"), "value": [1.0, 2.0]}
        )
    else:
        df = frame(
            {
                "h3_cell_id": pl.Series([CELL, None], dtype=pl.UInt64),
                "value": [1.0, 2.0],
            }
        )
    with pytest.raises(ValueError, match="Invalid H3"):
        H3View(df, vdims=["value"])


def test_mixed_resolutions_can_be_sliced(frame):
    df = frame(
        {
            "h3_cell_id": [CELL, h3.cell_to_parent(CELL, 3)],
            "h3_level": [4, 3],
            "value": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="mixed H3 resolutions"):
        H3View(df, vdims=["value"])
    view = H3View(df, vdims=["value"], coords=["h3_level"], selection={"h3_level": 3})
    assert view.h3_level == 3
    with pytest.raises(ValueError, match="does not match"):
        H3View(
            df,
            vdims=["value"],
            coords=["h3_level"],
            selection={"h3_level": 3},
            h3_level=4,
        )


def test_duplicates_are_not_aggregated(frame):
    df = frame({"h3_cell_id": [CELL, CELL], "value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="duplicate H3"):
        H3View(df, vdims=["value"])


def test_coordinate_errors_and_empty_selection(frame):
    df = frame({"h3_cell_id": [CELL, CELL], "week": [1, 2], "value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="multiple values"):
        H3View(df, vdims=["value"], coords=["week"])
    with pytest.raises(ValueError, match="declared in coords"):
        H3View(df, vdims=["value"], selection={"week": 1})
    with pytest.raises(TypeError, match="scalar"):
        H3View(df, vdims=["value"], coords=["week"], selection={"week": [1]})
    empty = H3View(df, vdims=["value"], coords=["week"], selection={"week": 99})
    assert empty.df.empty
    assert empty.h3_level is None
    assert np.isnan(empty._sample_view().data["value"]).all()


def test_nullable_values(frame):
    if frame is pd.DataFrame:
        df = frame(
            {"h3_cell_id": [CELL, OTHER], "value": pd.array([0, None], dtype="Int64")}
        )
    else:
        df = frame(
            {"h3_cell_id": [CELL, OTHER], "value": pl.Series([0, None], dtype=pl.Int64)}
        )
    view = H3View(df, vdims=["value"])
    assert view.df.loc[CELL, "value"] == 0
    assert np.isnan(view.df.loc[OTHER, "value"])


def test_null_coordinate(frame):
    df = frame(
        {"h3_cell_id": [CELL, OTHER], "species": [None, "raven"], "value": [1.0, 2.0]}
    )
    view = H3View(df, vdims=["value"], coords=["species"], selection={"species": None})
    assert list(view.df.index) == [CELL]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"vdims": []}, "at least one"),
        ({"vdims": "value"}, "sequence"),
        ({"vdims": ["missing"]}, "not found"),
        ({"vdims": ["value", "value"]}, "duplicate"),
        ({"variable": "missing"}, "included in vdims"),
        ({"coords": ["value"]}, "distinct"),
        ({"cell": "missing"}, "not found"),
        ({"nx": 0}, "positive integer"),
        ({"ny": 1.5}, "positive integer"),
        ({"h3_level": 16}, "between 0 and 15"),
    ],
)
def test_configuration_errors(kwargs, message):
    df = pd.DataFrame({"h3_cell_id": [CELL], "value": [1.0]})
    with pytest.raises((ValueError, TypeError), match=message):
        H3View(df, **({"vdims": ["value"]} | kwargs))


def test_nonnumeric_values(frame):
    df = frame({"h3_cell_id": [CELL], "value": ["1"]})
    with pytest.raises(TypeError, match="real numeric"):
        H3View(df, vdims=["value"])


def test_alias_and_assertion():
    df = pd.DataFrame({"custom": [CELL], "value": [1.0]})
    with pytest.warns(DeprecationWarning):
        view = H3View(df, h3_cell_column="custom", h3_level=4, vdims=["value"])
    assert view.cell == "custom"
    with pytest.raises(ValueError, match="disagree"):
        H3View(df, cell="custom", h3_cell_column="other", vdims=["value"])


def test_unsupported_inputs():
    for df in (None, {"value": [1]}, pl.DataFrame({"value": [1]}).lazy()):
        with pytest.raises(TypeError, match="eager"):
            H3View(df, vdims=["value"])

"""Select tabular coordinates and build a validated, single-resolution lookup."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from h3.api import basic_int as h3

_CELL_NAMES = (
    "h3_cell",
    "cell_id",
    "h3_cell_id",
    "h3_idx",
    "h3_cell_id_str",
)


@dataclass
class Slice:
    values: pd.DataFrame
    cell: str | None
    level: int | None
    selection: dict[str, Any]


def dimension_names(names: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(names, str):
        raise TypeError(f"{name} must be a sequence of column names, not a string.")
    result = tuple(names)
    if any(not isinstance(value, str) for value in result):
        raise TypeError(f"{name} must contain string column names.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate column names.")
    return result


def _cell_ids(values) -> np.ndarray:
    """Do not cast through floats: H3 integers exceed float64's exact range."""
    ids = np.empty(len(values), dtype=np.uint64)
    for i, value in enumerate(values):
        if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{15,16}", value):
            number = int(value, 16)
        elif isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
            number = int(value)
        else:
            raise ValueError(
                f"Invalid H3 cell at slice row {i}: {value!r}. "
                "Use integer or hexadecimal-string cell IDs; floats and nulls are invalid."
            )
        if not 0 <= number <= np.iinfo(np.uint64).max:
            raise ValueError(f"Invalid H3 cell at slice row {i}: {value!r}.")
        ids[i] = number
    for number in np.unique(ids):
        if not h3.is_valid_cell(int(number)):
            raise ValueError(f"Invalid H3 cell: {int(number):x}.")
    return ids


def _resolve_cell(df, cell: str | None, is_pandas: bool) -> tuple[str | None, bool]:
    if cell is not None:
        if cell in df.columns:
            return cell, False
        if is_pandas and df.index.nlevels == 1 and df.index.name == cell:
            return cell, True
        raise ValueError(f"Cell column {cell!r} was not found.")
    candidates = [name for name in _CELL_NAMES if name in df.columns]
    if len(candidates) == 1:
        return candidates[0], False
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous H3 columns {candidates!r}; specify cell= explicitly."
        )
    if is_pandas and df.index.nlevels == 1 and not isinstance(df.index, pd.RangeIndex):
        # Validate only the selected index below, just as for a selected column.
        return df.index.name, True
    raise ValueError("No H3 column was recognized; specify cell= explicitly.")


def snapshot_frame(df):
    """Own the source across future selections without modifying the caller's frame."""
    if isinstance(df, pd.DataFrame):
        if not df.columns.is_unique:
            raise ValueError("Dataframe column names must be unique.")
        return df.copy(deep=True)
    try:
        import polars as pl
    except ImportError:
        pl = None
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.clone()
    raise TypeError(
        "df must be an eager pandas or Polars DataFrame. "
        "Materialize queries and lazy frames before passing them to H3View."
    )


def coordinate_table(df, coords: tuple[str, ...]) -> pd.DataFrame:
    """Collect only observed coordinate combinations, never a Cartesian product."""
    missing = set(coords) - set(df.columns)
    if missing:
        raise ValueError(f"Coordinate columns not found: {sorted(missing)!r}.")
    if not coords:
        return pd.DataFrame()
    if isinstance(df, pd.DataFrame):
        return df.loc[:, list(coords)].drop_duplicates().reset_index(drop=True)
    unique = df.select(list(coords)).unique(maintain_order=True)
    return pd.DataFrame({name: unique[name].to_list() for name in coords})


def coordinate_values(table: pd.DataFrame, name: str) -> list[Any]:
    values = [None if pd.isna(value) else value for value in table[name].unique()]
    present = [value for value in values if value is not None]
    try:
        present = sorted(present)
    except TypeError:
        pass  # Mixed scalar types retain their order of appearance.
    return present + ([None] if None in values else [])


def _matching(table, name, value):
    mask = table[name].isna() if pd.isna(value) else table[name].eq(value)
    return table.loc[mask.fillna(False)]


def choose_selection(
    table: pd.DataFrame,
    coords: tuple[str, ...],
    previous: Mapping[str, Any],
    explicit: Mapping[str, Any],
) -> dict[str, Any]:
    """Honor explicit choices, retain compatible state, then choose valid defaults."""
    unknown = set(explicit) - set(coords)
    if unknown:
        raise ValueError(
            f"Selection keys must be declared in coords: {sorted(unknown)!r}."
        )
    for name, value in explicit.items():
        if not pd.api.types.is_scalar(value):
            raise TypeError(f"Selection for {name!r} must be a scalar.")
    explicit = {
        name: None if pd.isna(value) else value for name, value in explicit.items()
    }
    chosen = {name: value for name, value in previous.items() if name in coords}
    chosen.update(explicit)
    if table.empty:
        return chosen
    candidates = table
    for name, value in explicit.items():
        candidates = _matching(candidates, name, value)
    if candidates.empty:
        raise ValueError(
            f"No rows match the explicit coordinate selection {dict(explicit)!r}."
        )
    for name in coords:
        if name in explicit:
            continue
        retained = (
            _matching(candidates, name, chosen[name])
            if name in chosen
            else candidates.iloc[:0]
        )
        if retained.empty:
            chosen[name] = coordinate_values(candidates, name)[0]
            retained = _matching(candidates, name, chosen[name])
        candidates = retained
    return {
        name: None if pd.isna(candidates[name].iloc[0]) else candidates[name].iloc[0]
        for name in coords
    }


def select_data(
    df: Any,
    *,
    cell: str | None,
    vdims: tuple[str, ...],
    coords: tuple[str, ...],
    selection: Mapping[str, Any] | None,
    h3_level: int | None,
) -> Slice:
    is_pandas = isinstance(df, pd.DataFrame)
    pl = None
    if not is_pandas:
        try:
            import polars as pl
        except ImportError:
            pass
        if pl is None or not isinstance(df, pl.DataFrame):
            raise TypeError(
                "df must be an eager pandas or Polars DataFrame. "
                "Materialize DuckDB relations or lazy frames before passing them to H3View."
            )
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("Dataframe column names must be unique.")
    cell, use_index = _resolve_cell(df, cell, is_pandas)
    missing = set(vdims + coords) - set(df.columns)
    if missing:
        raise ValueError(f"Columns not found: {sorted(missing)!r}.")
    if set(vdims) & set(coords) or cell in vdims + coords:
        raise ValueError("Cell, value, and coordinate dimensions must be distinct.")
    chosen = dict(selection or {})
    unknown = set(chosen) - set(coords)
    if unknown:
        raise ValueError(
            f"Selection keys must be declared in coords: {sorted(unknown)!r}."
        )
    for name in coords:
        if name not in chosen:
            unique = df[name].unique()
            if len(unique) > 1:
                raise ValueError(
                    f"Coordinate {name!r} has multiple values; supply selection[{name!r}]."
                )
            if len(unique) == 1:
                chosen[name] = unique[0]
        if name in chosen and not pd.api.types.is_scalar(chosen[name]):
            raise TypeError(f"Selection for {name!r} must be a scalar.")

    columns = list(dict.fromkeys((() if use_index else (cell,)) + vdims))
    if is_pandas:
        mask = np.ones(len(df), dtype=bool)
        for name, value in chosen.items():
            matches = df[name].isna() if pd.isna(value) else df[name].eq(value)
            mask &= matches.fillna(False).to_numpy(dtype=bool)
        selected = df.loc[mask, columns]
        ids = _cell_ids(selected.index if use_index else selected[cell])
        for name in vdims:
            dtype = selected[name].dtype
            if not pd.api.types.is_numeric_dtype(
                dtype
            ) or pd.api.types.is_complex_dtype(dtype):
                raise TypeError(f"Value column {name!r} must be real numeric data.")
        arrays = {
            name: selected[name].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
            for name in vdims
        }
    else:
        predicates = []
        for name, value in chosen.items():
            if value is None or value is pd.NA or value is pd.NaT:
                predicate = pl.col(name).is_null()
            elif pd.isna(value):
                predicate = pl.col(name).is_nan() | pl.col(name).is_null()
            else:
                predicate = pl.col(name) == pl.lit(value)
            predicates.append(predicate)
        selected = (
            df.filter(*predicates).select(columns) if predicates else df.select(columns)
        )
        # Reading the integer series directly also preserves nullable UInt64 IDs.
        ids = _cell_ids(selected[cell])
        for name in vdims:
            dtype = selected.schema[name]
            if not (dtype.is_numeric() or dtype == pl.Boolean):
                raise TypeError(f"Value column {name!r} must be real numeric data.")
        arrays = {
            name: selected[name].cast(pl.Float64).to_numpy().copy() for name in vdims
        }

    index = pd.Index(ids, name=cell)
    if index.has_duplicates:
        raise ValueError(
            "The selected slice has duplicate H3 cells. Select additional coordinates "
            "or aggregate rows before constructing H3View."
        )
    levels = {h3.get_resolution(int(number)) for number in ids}
    if len(levels) > 1:
        raise ValueError(
            f"The selected slice contains mixed H3 resolutions {sorted(levels)}. "
            "Select one stored H3 level before rendering."
        )
    level = next(iter(levels), h3_level)
    if h3_level is not None and level != h3_level:
        raise ValueError(
            f"h3_level={h3_level} does not match the cells' resolution {level}."
        )
    return Slice(pd.DataFrame(arrays, index=index), cell, level, chosen)

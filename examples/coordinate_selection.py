# /// script
# requires-python = ">=3.13"
# dependencies = ["h3loviz[polars]", "marimo"]
# [tool.uv.sources]
# h3loviz = { path = "..", editable = true }
# ///

"""One persistent map, fed by marimo cells and automatic Panel controls."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import h3
    import holoviews as hv
    import marimo as mo
    import polars as pl

    from h3loviz import H3View

    hv.extension("bokeh")
    return H3View, h3, mo, pl


@app.cell
def _(H3View):
    view = H3View(vdims=["abundance", "checklists"], coords=["week"])
    view.show()
    return (view,)


@app.cell
def _(mo):
    source = mo.ui.dropdown(
        {"Dataset A: weeks 1 and 4": "a", "Dataset B: weeks 4 and 12": "b"},
        value="Dataset A: weeks 1 and 4",
        label="Source",
    )
    color = mo.ui.dropdown(
        ["abundance", "checklists"],
        value="abundance",
        label="Color from marimo",
    )
    week_four = mo.ui.run_button(label="Select week 4 from marimo")
    mo.vstack(
        [
            mo.md(
                "Switch datasets here; the map and its Panel controls stay in place. "
                "The marimo controls send updates but do not mirror subsequent Panel changes."
            ),
            mo.hstack([source, color, week_four]),
        ]
    )
    return color, source, week_four


@app.cell
def _(h3, pl, source, view):
    # Replace this synthetic preparation with file loading in a real notebook.
    _weeks, _scale = ([1, 4], 1.0) if source.value == "a" else ([4, 12], 2.0)
    _cells = sorted(h3.get_res0_cells())
    _df = pl.DataFrame(
        [
            {
                "h3_cell_id": cell,
                "week": week,
                "abundance": float((i + week) % 12) * _scale,
                "checklists": (i + 1) * 10,
            }
            for week in _weeks
            for i, cell in enumerate(_cells)
        ]
    )
    view.update(df=_df)


@app.cell
def _(color, view):
    view.update(select_vdim=color.value)


@app.cell
def _(view, week_four):
    if week_four.value:
        view.update(select_coords={"week": 4})


if __name__ == "__main__":
    app.run()

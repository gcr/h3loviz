# /// script
# requires-python = ">=3.13"
# dependencies = ["h3loviz[polars]", "marimo"]
# [tool.uv.sources]
# h3loviz = { path = "..", editable = true }
# ///

"""Synthetic data; the notebook owns controls and recreates the view per slice."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


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
def _(h3, pl):
    # Broad synthetic coverage makes the initial world view visible.
    cells = sorted(h3.get_res0_cells())
    data = pl.DataFrame(
        [
            {
                "h3_cell_id": cell,
                "week": week,
                "abundance": float((i + week) % 12),
                "checklists": (i + 1) * 10,
            }
            for week in [1, 2, 3]
            for i, cell in enumerate(cells)
        ]
    )
    return (data,)


@app.cell
def _(mo):
    week = mo.ui.slider(1, 3, value=2, label="Week", show_value=True)
    variable = mo.ui.dropdown(
        ["abundance", "checklists"],
        value="abundance",
        label="Color by",
    )
    mo.hstack([week, variable])
    return variable, week


@app.cell
def _(H3View, data, variable, week):
    view = H3View(
        data,
        vdims=["abundance", "checklists"],
        coords=["week"],
        selection={"week": week.value},
        variable=variable.value,
    )
    view.show()


if __name__ == "__main__":
    app.run()

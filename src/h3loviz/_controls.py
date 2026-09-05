"""Panel controls generated from observed HoloViews dimension values."""

from datetime import date

import holoviews as hv
import numpy as np
import panel as pn

from ._data import coordinate_values


def _widget(name, label, options, current):
    values = options or [current]
    if any(isinstance(value, (date, np.datetime64)) for value in values):
        # HoloViews coerces date dimension values to datetime64 while cloning,
        # which can invalidate a Python date default. Preserve source scalars.
        widget_type = pn.widgets.Select if None in values else pn.widgets.DiscreteSlider
        widget = widget_type(
            label=label, options={str(value): value for value in values}, value=current
        )
    else:
        dim = hv.Dimension(name, label=label, values=values, default=current)
        metadata = hv.DynamicMap(lambda **kwargs: hv.Curve([]), kdims=[dim])
        widgets, _ = pn.pane.HoloViews.widgets_from_dimensions(metadata)
        widget = widgets[0]
    widget.param.update(disabled=len(options) <= 1, width=210, margin=(5, 0))
    return widget


def make_controls(coords, vdims, select_coords, select_vdim, table):
    """Prepare widgets without connecting callbacks or changing an existing view."""
    controls = []
    for i, name in enumerate(coords):
        options = coordinate_values(table, name) if name in table else []
        controls.append(
            (name, _widget(f"coord_{i}", name, options, select_coords.get(name)))
        )
    if vdims:
        controls.append((None, _widget("color", "Color by", list(vdims), select_vdim)))
    return controls

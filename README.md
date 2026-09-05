# h3loviz -- scalable visualization of H3 dataframes in the Holoviz ecosystem

This library renders dataframes (polars,pandas,duckdb,...) with H3 cell indices as interactive visualizations. Other tools like [lonboard](https://developmentseed.org/lonboard/latest/) and [xdggs.explore()](https://github.com/xarray-contrib/xdggs) render each H3 cell as a polygon, which can be slow when you have millions of unique H3 cells.

To get around this limitation, we use holoviz to render H3 cells as a rasterized image using datashader, which is much faster and more scalable. This library is designed to work with the Holoviz ecosystem, including [panel](https://panel.holoviz.org/), [holoviews](https://holoviews.org/), and [datashader](https://datashader.org/).


# When to use
- *Do you have raster data?* You don't want this library; use xarray for data management and geovis/datashader/... for visualization.
- *Do you have vector or shape data?* You don't want this library; use geopandas/shapely/...
- *Do you have a columnar dataframe with H3 cell indices? Do you need to do additional filtering/analysis?* This is the library for you!

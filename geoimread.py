# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:58:22 2019

@author: ag
"""

import xarray as xr
import numpy as np
import rasterio
from rasterio.warp import transform


def geoimread(fname, roi_x=None, roi_y=None, roi_crs=None, buffer=0, band=0):
    """Reads a sub-region of a geotiff/geojp2/ that overlaps a region of interest (roi)

    This is a simple wrapper of xarray functionality.

    Parameters:
    fname (str) : file.
    roi_x, roi_y (List[float]) : region of interest.
    roi_crs (dict) : ROI coordinate reference system, in rasterio dict format (if different from scene coordinates)
    buffer (float) : adds a buffer around the region of interest.
    band (int) : which bands to read.

    Returns
        xr.core.dataarray.DataArray : The cropped scene.

    """
    da = xr.open_rasterio(fname)
    if roi_x is not None:
        if not hasattr(roi_x, "__len__"):
            roi_x = [roi_x]
            roi_y = [roi_y]
        if roi_crs is not None:
            if str(roi_crs) == "LL":
                roi_crs = {"init": "EPSG:4326"}
            # remove nans
            roi_x = np.array(roi_x)
            roi_y = np.array(roi_y)
            ix = ~np.isnan(roi_x + roi_y)
            roi_x, roi_y = transform(
                src_crs=roi_crs, dst_crs=da.crs, xs=roi_x[ix], ys=roi_y[ix]
            )
        rows = (da.y > np.min(roi_y) - buffer) & (da.y < np.max(roi_y) + buffer)
        cols = (da.x > np.min(roi_x) - buffer) & (da.x < np.max(roi_x) + buffer)
        # update transform property
        da = da[band, rows, cols].squeeze()
        da.attrs["transform"] = (
            da.x.values[1] - da.x.values[0],
            0.0,
            da.x.values[0],
            0.0,
            da.y.values[1] - da.y.values[0],
            da.y.values[0],
        )
        return da
    else:
        return da[band, :, :].squeeze()


def geoimwrite(da, output_filename, compression="LZW", tiled=True, predictor=2):
    """... TODO ...
    """
    da = da.load()
    shape = da.shape
    bands = np.arange(shape[0]) + 1
    if len(da.shape) == 2:
        shape = np.insert(shape, 0, 1)
        bands = 1
    T = da.attrs["transform"]

    T = (T[2], T[0], T[1], T[5], T[3], T[4])  # WHY!!!!! - is this always the cas
    print(T)
    with rasterio.open(
        output_filename,
        "w",
        driver="GTiff",
        height=shape[1],
        width=shape[2],
        dtype=str(da.dtype),
        count=shape[0],
        compress=compression,
        tiled=tiled,
        predictor=predictor,
        crs=rasterio.crs.CRS.from_string(da.attrs["crs"]),
        transform=rasterio.Affine.from_gdal(*T),
    ) as dst:
        dst.write(da.values, bands)


if __name__ == "__main__":
    # test code...
    # Read the data
    fA = "https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/023/001/LC08_L1TP_023001_20150708_20170407_01_T1/LC08_L1TP_023001_20150708_20170407_01_T1_B8.TIF"
    A = geoimread(
        fA,
        roi_x=[-30.19, np.nan],
        roi_y=[81.245, np.nan],
        roi_crs={"init": "EPSG:4326"},
        buffer=20000,
    )

    import matplotlib.pyplot as plt

    ax = plt.axes()
    A.plot.imshow(cmap="gray", add_colorbar=False)
    ax.set_aspect("equal")
    ax.autoscale(tight=True)


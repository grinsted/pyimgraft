# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:58:22 2019

@author: ag
"""

import xarray as xr
import numpy as np
from rasterio.warp import transform

def geoimread(fname, roi_x=None, roi_y=None, roi_crs=None, buffer=0.0, band=1):
    da = xr.open_rasterio(fname)
    if roi_x is not None:
        if not hasattr(roi_x, "__len__"):
            roi_x = [roi_x]
            roi_y = [roi_y]
        if roi_crs is not None:
            roi_x,roi_y = transform(src_crs=roi_crs, dst_crs=da.crs, xs=roi_x, ys=roi_y)
        rows = (da.y > np.min(roi_y) - buffer) & (da.y < np.max(roi_y) + buffer)
        cols = (da.x > np.min(roi_x) - buffer) & (da.x < np.max(roi_x) + buffer)
        return da[band-1,rows,cols].squeeze()
    else:
        return da[band-1,:,:].squeeze()
    
if __name__ == "__main__":
    # Read the data
    fA = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/023/001/LC08_L1TP_023001_20150708_20170407_01_T1/LC08_L1TP_023001_20150708_20170407_01_T1_B8.TIF'
    A = geoimread(fA, roi_x=-30.19, roi_y=81.245, roi_crs={'init': 'EPSG:4326'}, buffer=20000)

    import matplotlib.pyplot as plt
    ax = plt.axes()
    A.plot.imshow(cmap='gray', add_colorbar=False)
    ax.set_aspect('equal')
    ax.autoscale(tight=True)

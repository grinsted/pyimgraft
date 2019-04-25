# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:58:22 2019

@author: ag
"""

import xarray as xr
import numpy as np
from rasterio.warp import transform

def geoimread(fname, x=None, y=None, roi_crs=None, buffer=0.0, band=1):
    da = xr.open_rasterio(fname)
    if x is not None:
        if not hasattr(x, "__len__"):
            x = [x]
            y = [y]
        if roi_crs is not None:
            x,y = transform(src_crs=roi_crs, dst_crs=da.crs, xs=x, ys=y)
        rows = (da.y > np.min(y) - buffer) & (da.y < np.max(y) + buffer)
        cols = (da.x > np.min(x) - buffer) & (da.x < np.max(x) + buffer)
        return da[band-1,rows,cols].squeeze()
    else:
        return da[band-1,:,:].squeeze()
    
# Read the data
fA = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/023/001/LC08_L1TP_023001_20150708_20170407_01_T1/LC08_L1TP_023001_20150708_20170407_01_T1_B8.TIF'
fB = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/023/001/LC08_L1TP_023001_20160710_20170323_01_T1/LC08_L1TP_023001_20160710_20170323_01_T1_B8.TIF'

A = geoimread(fA, x=-30.19, y=81.245, roi_crs={'init': 'EPSG:4326'}, buffer=20000)
B = geoimread(fB, x=-30.19, y=81.245, roi_crs={'init': 'EPSG:4326'}, buffer=20000)

A.plot.imshow()


#da = xr.open_rasterio(fA)
#db = xr.open_rasterio(fB)
#
#wgs84 = {'init': 'EPSG:4326'}
#p = (-30.19, 81.245)
#x0,y0 = transform(src_crs=wgs84, dst_crs=da.crs, xs=[p[0]], ys=[p[1]])
#buffer=2000
#
#cols = (da.x > x0[0]-buffer ) & (da.x < x0[0]+buffer)
#rows = (da.y > y0[0]-buffer ) & (da.y < y0[0]+buffer)
#
#A = da[0,rows,cols]
#A.plot.imshow()


#da[:, 400:600, 400:600].plot.imshow()


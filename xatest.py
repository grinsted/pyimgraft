# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:58:22 2019

@author: ag
"""

import xarray as xr

# Read the data
da = xr.open_rasterio('https://github.com/mapbox/rasterio/raw/master/tests/data/RGB.byte.tif')
da[:, 400:600, 400:600].plot.imshow()


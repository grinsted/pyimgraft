# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 13:44:50 2018

@author: aslak
"""


import rasterio
from affine import Affine
from pyproj import Proj, transform
import numpy as np

import matplotlib.pyplot as plt


class Geoimage:
    def __init__(self,data,x,y,affine,projection,extent,meta={}):
        self.data = data
        self.x = x
        self.y = y
        self.affine = affine
        self.projection = projection
        self.extent = extent
        self.meta = meta
        
    def plot(self):
        plt.imshow(self.data,extent=self.extent, cmap='gray')


def geoimread(fname,lon=None,lat=None,padding=0.0,band=1):
    
    if np.isscalar(padding):
        padding = [padding, padding]
    
    with rasterio.open(fname) as src:
        Ill = Proj(proj='latlon',datum='WGS84')
        Imap = Proj(src.crs) 
        if not lat is None:
            if np.any(abs(lon)>360) or np.any(abs(lat)>90):
                cx = lon
                cy = lat
            else:
                cx,cy = transform(Ill,Imap,lon,lat) #lat lon flipped

            
            corner1 = ~src.transform * (np.min(cx)-padding[0], np.min(cy)-padding[1])
            corner2 = ~src.transform * (np.max(cx)+padding[0], np.max(cy)+padding[1])
            cols = sorted([corner1[0], corner2[0]])
            rows = sorted([corner1[1], corner2[1]])
        else:
            cols=[0, np.Inf]
            rows=[0, np.Inf]
            cx=np.mean([src.bounds.left, src.bounds.right])
            cy=np.mean([src.bounds.top, src.bounds.bottom])

        cols = np.round(np.clip(cols,0,src.width)).tolist()
        rows = np.round(np.clip(rows,0,src.height)).tolist()
        A = src.read(band, window=(rows, cols))
        x = np.arange(cols[0],cols[1])
        y = np.arange(rows[0],rows[1])
        for ix,xx in enumerate(x):
            x[ix] = (src.transform * (xx,np.mean(cy)))[0]
        for ix,yy in enumerate(y):
            y[ix] = (src.transform * (np.mean(cx),yy))[1]
        newaffine =  src.transform * Affine.translation(cols[0],rows[0])            
        extent = (x[0],x[-1],y[-1],y[0])
        meta = {'crs': src.crs, 'nodataval': src.nodatavals[band-1]}
        #do not use meta from src directly as it is not consistent with subset. 
        if A.dtype.kind == 'f': #these types support nan!
            A[A == meta['nodataval']] = np.nan
    return Geoimage(A,x,y,newaffine,Imap,extent,meta)


if __name__ == "__main__":
    
    fA= 'https://storage.googleapis.com/gcp-public-data-landsat/LT05/01/023/001/LT05_L1TP_023001_19940714_20170113_01_T2/LT05_L1TP_023001_19940714_20170113_01_T2_B3.TIF'
    A = geoimread(fA,-30.19,81.245,10000)
    A.plot()
    
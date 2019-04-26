pyimgraft
======

Initial work to bring imgraft to python. -Or atleast the feature tracker.






Example
==========


```python
from templatematch import templatematch
from matplotlib import pyplot as plt
from geoimread import geoimread

fA = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/023/001/LC08_L1TP_023001_20150708_20170407_01_T1/LC08_L1TP_023001_20150708_20170407_01_T1_B8.TIF'
fB = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/023/001/LC08_L1TP_023001_20160710_20170323_01_T1/LC08_L1TP_023001_20160710_20170323_01_T1_B8.TIF'

# Use geoimread/xarray to read a tiny region of a cloud optimized geotif.
A = geoimread(fA, roi_x=-30.19, roi_y=81.245, roi_crs={'init': 'EPSG:4326'}, buffer=20000)
B = geoimread(fB, roi_x=-30.19, roi_y=81.245, roi_crs={'init': 'EPSG:4326'}, buffer=20000)

# Do the feature tracking. 
r = templatematch(A, B, TemplateWidth=128, SearchWidth=128 + 64)

# remove outliers (excessive local strain) or very low snr
r.clean()

#show image A
ax = plt.axes()
A.plot.imshow(cmap='gray', add_colorbar=False)
ax.set_aspect('equal')
ax.autoscale(tight=True)

# drape displacement results over image A. 
r.plot(x=A.x, y=A.y)  


```


Dependencies
==============
* pyfftw
* xarray (rasterio)
* pyproj

```python
conda install pyfftw -c conda-forge
conda install rasterio pyproj xarray
```
You may need to set the GDAL_DATA environment variable to get it to work.

TODO
=======
This is not finished

* Make better interfaces (explore using xarray instead of geoimage)
* Make more tests
* setup a build bot 
* Separate plotting from return data? Figure out how to tie match result to projection (again xarray would be useful)


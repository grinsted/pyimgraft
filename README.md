pyimgraft
======

Initial work to bring imgraft to python. -Or atleast the feature tracker.






Example
==========


```python

from geoimread import geoimread
from templatematch import templatematch

fA='https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/023/001/LC08_L1TP_023001_20150708_20170407_01_T1/LC08_L1TP_023001_20150708_20170407_01_T1_B8.TIF'
fB='https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/023/001/LC08_L1TP_023001_20160710_20170323_01_T1/LC08_L1TP_023001_20160710_20170323_01_T1_B8.TIF'

A = geoimread(fA,-30.19,81.245,20000)
B = geoimread(fB,-30.19,81.245,20000)

A.plot()



r=templatematch(A.data,B.data,TemplateWidth=128,SearchWidth=128+64)

r.clean() #remove dodgy points (excessive local strain or poor signal to noise)
r.plot(x=A.x, y=A.y)
plt.clim([0, 3])
plt.colorbar()

```


Dependencies
==============
* pyfftw
* rasterio (and some of its dependencies)
* pyproj

```python
conda install pyproj -c conda-forge
conda install rasterio pyproj
```

TODO
=======
This is not finished

* Make better interfaces (explore using xarray instead of geoimage)
* Make more tests
* setup a build bot 
* Separate plotting from return data? Figure out how to tie match result to projection (again xarray would be useful)


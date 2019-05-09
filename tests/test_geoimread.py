# -*- coding: utf-8 -*-

import pytest

import pyimgraft



def test_readfromcloud():
    fA = "https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/023/001/LC08_L1TP_023001_20150708_20170407_01_T1/LC08_L1TP_023001_20150708_20170407_01_T1_B8.TIF"
    A = pyimgraft.geoimread(
        fA, roi_x=[-30.19], roi_y=[81.245], roi_crs={"init": "EPSG:4326"}, buffer=200
    )
    print(A.shape)
    assert A.shape == (27,26)
    assert A.x.shape == (26,)
    assert float(A.x[0]) == 547545

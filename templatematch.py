

import numpy as np
import scipy.ndimage
from scipy.signal import medfilt2d
from scipy.interpolate import interp1d
import pyfftw
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'


class MatchResult:
    def __init__(self, pu, pv, du, dv, peakCorr, meanAbsCorr, method):
        self.pu = pu
        self.pv = pv
        self.du = du
        self.dv = dv
        self.peakCorr = peakCorr
        self.meanAbsCorr = meanAbsCorr
        self.snr = peakCorr / meanAbsCorr
        self.method = method

    def clean(self, maxstrain=0.1, minsnr=1.02):
        # assume that pu and pv are arranged in a regular grid.
        # TODO: input checking...
        resolution = np.abs(self.pu[1, 1] - self.pu[0, 0])  # assume constant resolution in all directions...
        strain = np.sqrt((self.du - medfilt2d(self.du)) ** 2 + (self.dv - medfilt2d(self.dv)) ** 2) / resolution
        ix = np.logical_or(strain > maxstrain, self.snr < minsnr)  # skip nans to avoid warnings
        self.du[ix] = np.nan
        self.dv[ix] = np.nan

    def plot(self, x=None, y=None, alpha=0.7):
        px = self.pu
        py = self.pv
        dx = 1.0
        dy = 1.0
        if x is not None:
            px = interp1d(np.arange(0, x.shape[0]), x, fill_value='extrapolate')(px)
            dx = x[1] - x[0]
        if y is not None:
            py = interp1d(np.arange(0, y.shape[0]), y, fill_value='extrapolate')(py)
            dy = y[1] - y[0]

        C = np.sqrt(self.du ** 2 + self.dv ** 2)

        plt.pcolormesh(get_corners(px), get_corners(py), C, alpha=alpha)
        plt.colorbar()
        plt.quiver(px, py, self.du * dx, self.dv * dy)

    # TODO add more methods to plot and clean results....


def get_corners(pu):
    # helper function to generate inputs for pcolormesh

    # extend longitude by 1
    pu_extend = np.zeros((pu.shape[0] + 2, pu.shape[1] + 2))
    pu_extend[1:-1, 1:-1] = pu  # fill up with original values
    # fill in extra endpoints
    pu_extend[:, 0] = pu_extend[:, 1] + (pu_extend[:, 1] - pu_extend[:, 2])
    pu_extend[:, -1] = pu_extend[:, -2] + (pu_extend[:, -2] - pu_extend[:, -3])
    pu_extend[0, :] = pu_extend[1, :] + (pu_extend[1, :] - pu_extend[2, :])
    pu_extend[-1, :] = pu_extend[-2, :] + (pu_extend[-2, :] - pu_extend[-3, :])
    # calculate the corner points
    # return scipy.signal.convolve2d(pu_extend, np.ones((2, 2)/4), mode='valid')  # TODO: remove dependency
    return (pu_extend[0:-2, 0:-2] + pu_extend[0:-2, 1:-1] + pu_extend[1:-1, 0:-2] + pu_extend[1:-1, 1:-1]) / 4.0


def templatematch(A, B, pu=None, pv=None, TemplateWidth=128, SearchWidth=128 + 16, Initialdu=0, Initialdv=0):
    """Feature tracking by template matching

    Usage : r = templatematch(A, B, TemplateWidth=64, SearchWidth=128)

    Notes : Currently only orientation correlation is implemented.

    Parameters
    ----------
    A, B : array_like
        Two images as 2d-matrices.
    pu, pv : array_like, optional
        Pixel coordinates in image A that you would like to find in image B (default is to drape a grid over A)
    TemplateWidth : int, optional
        pixel-size of the small templates being cut from image A.
    SearchWidth : int, optional
        pixel-size of the search region within image B.
    Initialdu, Initialdv : int, optional
        An initial guess of the displacement. The search window will be offset by this. (default = 0)

    Returns:
    ----------
        result : MatchResult

    """
    if not np.any(np.iscomplex(A)):  # always do Orientation correlation!
        A = forient(A)
        B = forient(B)

    if pu is None:
        pu = np.arange(SearchWidth / 2, A.shape[1] - SearchWidth / 2 + TemplateWidth / 2, TemplateWidth / 2)
        pv = np.arange(SearchWidth / 2, A.shape[0] - SearchWidth / 2 + TemplateWidth / 2, TemplateWidth / 2)
        pu, pv = np.meshgrid(pu, pv)

    du = np.full(pu.shape, np.nan)  # np.empty(pu.shape) * np.nan
    dv = np.full(pu.shape, np.nan)  #
    peakCorr = np.full(pu.shape, np.nan)  #
    meanAbsCorr = np.full(pu.shape, np.nan)  #
    if np.isscalar(Initialdu):
        Initialdu = np.zeros(pu.shape) + Initialdu
    if np.isscalar(Initialdv):
        Initialdv = np.zeros(pu.shape) + Initialdv

    if np.any(np.iscomplex(B)):
        B = np.conj(B)

    SearchHeight = SearchWidth  # TODO: Clean-up. Dont support heights in python version.
    TemplateHeight = TemplateWidth

    # PREPARE PYFFTW:
    # --------preallocate arrays for pyfftw ---------------
    AArot90 = pyfftw.empty_aligned((TemplateWidth, TemplateHeight), dtype='complex64', order='C', n=None)
    BB = pyfftw.empty_aligned((SearchHeight, SearchWidth), dtype='complex64', order='C', n=None)
    CC_sz = np.add(AArot90.shape, BB.shape) - 1
    CC = pyfftw.empty_aligned(CC_sz, dtype='complex64', order='C', n=None)
    # fT = pyfftw.empty_aligned(sz, dtype='complex64', order='C', n=None)
    # fB = pyfftw.empty_aligned(sz, dtype='complex64', order='C', n=None)
    #
    fft2AA = pyfftw.builders.fft2(AArot90, s=CC_sz, overwrite_input=True, auto_contiguous=True)
    fft2BB = pyfftw.builders.fft2(BB, s=CC_sz, overwrite_input=True, auto_contiguous=True)
    ifft2CC = pyfftw.builders.ifft2(CC, overwrite_input=True, auto_contiguous=True, avoid_copy=True)

    # precalculate how to interpret CC
    wkeep = np.subtract(BB.shape, AArot90.shape) / 2  # cut away edge effects
    C_center = (CC_sz - 1) / 2  # center
    C_rows = ((C_center[0] - wkeep[0]).astype('int'), (C_center[0] + wkeep[0]).astype('int'))
    C_cols = ((C_center[1] - wkeep[1]).astype('int'), (C_center[1] + wkeep[1]).astype('int'))
    C_uu = np.arange(-wkeep[1], wkeep[1] + 1)
    C_vv = np.arange(-wkeep[0], wkeep[0] + 1)
    # -----------------------------------------------------

    for ii, u in np.ndenumerate(pu):
        p = np.array([u, pv[ii]])

        initdu = Initialdu[ii]
        initdv = Initialdv[ii]
        # Actual pixel centre might differ from (pu, pv) because of rounding
        #
        Acenter = np.round(p) - (TemplateWidth / 2 % 1)
        Bcenter = np.round(p + np.array([initdu, initdv])) - (SearchWidth / 2 % 1)  # centre coordinate of search region

        # we should return coords that was actually used:
        pu[ii] = Acenter[0]
        pv[ii] = Acenter[1]
        initdu = Bcenter[0] - Acenter[0]  # actual offset
        initdv = Bcenter[1] - Acenter[1]

        try:
            Brows = (Bcenter[1] + (-SearchHeight / 2, SearchHeight / 2)).astype('int')  # TODO: check "+1"
            Bcols = (Bcenter[0] + (-SearchWidth / 2, SearchWidth / 2)).astype('int')
            Arows = (Acenter[1] + (-TemplateHeight / 2, TemplateHeight / 2)).astype('int')
            Acols = (Acenter[0] + (-TemplateWidth / 2, TemplateWidth / 2)).astype('int')
            if Brows[0] < 0 or Arows[0] < 0 or Bcols[0] < 0 or Acols[0] < 0:
                continue
            if Brows[1] >= B.shape[0] or Arows[1] >= A.shape[0] or Bcols[1] >= B.shape[1] or Acols[1] >= A.shape[1]:
                continue  # handled by exception
            BB[:, :] = B[Brows[0]:Brows[1], Bcols[0]:Bcols[1]]
            AArot90[:, :] = np.rot90(A[Arows[0]:Arows[1], Acols[0]:Acols[1]], 2)
        except IndexError:
            continue

        # --------------- CCF ------------------
        fT = fft2AA(AArot90)
        fB = fft2BB(BB)
        fT[:] = np.multiply(fB, fT)
        CC = np.real(ifft2CC(fT))

        C = CC[C_rows[0]:C_rows[1], C_cols[0]:C_cols[1]]

        # --------------------------------------

        mix = np.unravel_index(np.argmax(C), C.shape)
        Cmax = C[mix[0], mix[1]]
        meanAbsCorr[ii] = np.mean(abs(C))
        edgedist = np.min([mix, np.subtract(C.shape, mix) - 1])
        if edgedist == 0:
            continue  # because we dont trust peak if at edge of domain.
        else:
            ww = np.amin((edgedist, 6))
            c = C[mix[0] - ww:mix[0] + ww + 1, mix[1] - ww:mix[1] + ww + 1]
            [uu, vv] = np.meshgrid(C_uu[mix[1] - ww:mix[1] + ww + 1], C_vv[mix[0] - ww:mix[0] + ww + 1])

            # simple, fast, and excellent performance for landsat test images.
            c = c - np.mean(np.abs(c.ravel()))
            c[c < 0] = 0
            c = c / np.sum(c)
            mix = (np.sum(np.multiply(vv, c)), np.sum(np.multiply(uu, c)))
        du[ii] = mix[1] + initdu
        dv[ii] = mix[0] + initdv
        peakCorr[ii] = Cmax
    return MatchResult(pu, pv, du, dv, peakCorr, meanAbsCorr, method='OC')


def forient(img):
    f = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    r = scipy.ndimage.convolve(img, f, mode='nearest').astype(np.complex64)
    r = r + complex(0, 1) * scipy.ndimage.convolve(img, np.rot90(f), mode='nearest')
    m = np.abs(r)
    m[m == 0] = 1
    r = np.divide(r, m)

    return r


if __name__ == "__main__":
    from geoimread import geoimread
    import matplotlib.pyplot as plt
    # from skimage.transform import rescale, resize, downscale_local_mean

    # fA= 'https://storage.googleapis.com/gcp-public-data-landsat/LT05/01/023/001/LT05_L1TP_023001_19940714_20170113_01_T2/LT05_L1TP_023001_19940714_20170113_01_T2_B3.TIF'
    # fB= 'https://storage.googleapis.com/gcp-public-data-landsat/LT05/01/023/001/LT05_L1TP_023001_19940916_20170112_01_T2/LT05_L1TP_023001_19940916_20170112_01_T2_B3.TIF'

    fA = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/023/001/LC08_L1TP_023001_20150708_20170407_01_T1/LC08_L1TP_023001_20150708_20170407_01_T1_B8.TIF'
    fB = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/023/001/LC08_L1TP_023001_20160710_20170323_01_T1/LC08_L1TP_023001_20160710_20170323_01_T1_B8.TIF'

    A = geoimread(fA, -30.19, 81.245, 20000)
    B = geoimread(fB, -30.19, 81.245, 20000)

    import time
    time1 = time.time()
    r = templatematch(A.data, B.data, TemplateWidth=128, SearchWidth=128 + 64)
    time2 = time.time()

    plt.figure()
    from matplotlib import pyplot as plt
    A.plot()

    r.clean()
    r.plot(x=A.x, y=A.y)
    print((time2 - time1) * 1000.0)
    # plt.hist(r.du.ravel())

    print(np.nanmean(r.du.ravel()))
    print(np.nanmean(r.dv.ravel()))

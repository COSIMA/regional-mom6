from numba import njit
import numpy as np
from dask.base import tokenize
import dask.array as dsa
import xarray as xr


def flood_kara(data, xdim='lon', ydim='lat', zdim='z', tdim='time',
               spval=1e+15):
    """Apply extrapolation onto land from Kara algo.
    Arguments:
        data {xarray.DataArray} -- input data
    Keyword Arguments:
        xdim {str} -- name of x dimension (default: {'lon'})
        ydim {str} -- name of y dimension (default: {'lat'})
        zdim {str} -- name of z dimension (default: {'z'})
        tdim {str} -- name of time dimension (default: {'time'})
        spval {float} -- missing value (default: {1e+15})
    Returns:
        xarray.DataArray -- result of the extrapolation
    """
    # check for input data shape
    if tdim not in data.dims:
        data = data.expand_dims(dim=tdim)
    if zdim not in data.dims:
        data = data.expand_dims(dim=zdim)

    nrec = len(data[tdim])
    nlev = len(data[zdim])
    ny = len(data[ydim])
    nx = len(data[xdim])
    shape = (nrec, nlev, ny, nx)
    chunks = (1, 1, ny, nx)

    def compute_chunk(zlev, trec):
        data_slice = data.isel({tdim: trec, zdim: zlev})
        return flood_kara_xr(data_slice, spval=spval)[None, None]

    name = str(data.name) + '-' + tokenize(data.name, shape)
    dsk = {(name, rec, lev, 0, 0,): (compute_chunk, lev, rec)
           for lev in range(nlev)
           for rec in range(nrec)}

    out = dsa.Array(dsk, name, chunks,
                    dtype=data.dtype, shape=shape)

    xout = xr.DataArray(data=out, name=str(data.name),
                        coords={tdim: data[tdim],
                                zdim: data[zdim],
                                ydim: data[ydim],
                                xdim: data[xdim]},
                        dims=(tdim, zdim, ydim, xdim))

    # rechunk the result
    xout = xout.chunk({tdim: 1, zdim: nlev, ydim: ny, xdim: nx})

    return xout


def flood_kara_xr(dataarray, spval=1e+15):
    """Apply flood_kara on a xarray.dataarray
    Arguments:
        dataarray {xarray.DataArray} -- input 2d data array
    Keyword Arguments:
        spval {float} -- missing value (default: {1e+15})
    Returns:
        numpy.ndarray -- field after extrapolation
    """

    masked_array = dataarray.squeeze().to_masked_array()
    out = flood_kara_ma(masked_array, spval=spval)
    return out


def flood_kara_ma(masked_array, spval=1e+15):
    """Apply flood_kara on a numpy masked array
    Arguments:
        masked_array {np.ma.masked_array} -- array to extrapolate
    Keyword Arguments:
        spval {float} -- missing value (default: {1e+15})
    Returns:
        out -- field after extrapolation
    """

    field = masked_array.data

    if np.isnan(field).all():
        # all the values are NaN, can't do anything
        out = field.copy()
    else:
        # proceed with extrapolation
        field[np.isnan(field)] = spval
        mask = np.ones(field.shape)
        mask[masked_array.mask] = 0
        out = flood_kara_raw(field, mask)
    return out


@njit
def flood_kara_raw(field, mask, nmax=1000):
    """Extrapolate land values onto land using the kara method
    (https://doi.org/10.1175/JPO2984.1)
    Arguments:
        field {np.ndarray} -- field to extrapolate
        mask {np.ndarray} -- land/sea binary mask (0/1)
    Keyword Arguments:
        nmax {int} -- max number of iteration (default: {1000})
    Returns:
        drowned -- field after extrapolation
    """

    ny, nx = field.shape
    nxy = nx * ny
    # create fields with halos
    ztmp = np.zeros((ny+2, nx+2))
    zmask = np.zeros((ny+2, nx+2))
    # init the values
    ztmp[1:-1, 1:-1] = field.copy()
    zmask[1:-1, 1:-1] = mask.copy()

    ztmp_new = ztmp.copy()
    zmask_new = zmask.copy()
    #
    nt = 0
    while (zmask[1:-1, 1:-1].sum() < nxy) and (nt < nmax):
        for jj in np.arange(1, ny+1):
            for ji in np.arange(1, nx+1):

                # compute once those indexes
                jjm1 = jj-1
                jjp1 = jj+1
                jim1 = ji-1
                jip1 = ji+1

                if (zmask[jj, ji] == 0):
                    c6 = 1 * zmask[jjm1, jim1]
                    c7 = 2 * zmask[jjm1, ji]
                    c8 = 1 * zmask[jjm1, jip1]

                    c4 = 2 * zmask[jj, jim1]
                    c5 = 2 * zmask[jj, jip1]

                    c1 = 1 * zmask[jjp1, jim1]
                    c2 = 2 * zmask[jjp1, ji]
                    c3 = 1 * zmask[jjp1, jip1]

                    ctot = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8

                    if (ctot >= 3):
                        # compute the new value for this point
                        zval = (c6 * ztmp[jjm1, jim1] +
                                c7 * ztmp[jjm1, ji] +
                                c8 * ztmp[jjm1, jip1] +
                                c4 * ztmp[jj, jim1] +
                                c5 * ztmp[jj, jip1] +
                                c1 * ztmp[jjp1, jim1] +
                                c2 * ztmp[jjp1, ji] +
                                c3 * ztmp[jjp1, jip1]) / ctot

                        # update value in field array
                        ztmp_new[jj, ji] = zval
                        # set the mask to sea
                        zmask_new[jj, ji] = 1
        nt += 1
        ztmp = ztmp_new.copy()
        zmask = zmask_new.copy()

        if nt == nmax:
            raise ValueError('number of iterations exceeded maximum, '
                             'try increasing nmax')

    drowned = ztmp[1:-1, 1:-1]

    return drowned
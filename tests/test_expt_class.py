import numpy as np
import pytest
from regional_mom6 import experiment

import xarray as xr
# reload(experiment)

@pytest.mark.parametrize(
(       "xextent",
        "yextent",
        "daterange",
        "resolution",
        "vlayers",
        "dz_ratio",
        "depth",
        "mom_run_dir",
        "mom_input_dir",
        "toolpath",
        "gridtype"
),
    [
        ([-5,5],[0,10],
         ["2003-01-01 00:00:00","2003-01-01 00:00:00"],
         0.1,
         5,
         1,
         1000,
         "rundir",
         "inputdir",
         "toolpath",
         "even_spacing"),
    ],
)
def test_experiment(
    xextent,
    yextent,
    daterange,
    resolution,
    vlayers,
    dz_ratio,
    depth,
    mom_run_dir,
    mom_input_dir,
    toolpath,
    gridtype):

    expt = experiment(
        xextent,
        yextent,
        daterange,
        resolution,
        vlayers,
        dz_ratio,
        depth,
        mom_run_dir,
        mom_input_dir,
        toolpath,
        gridtype
    )

    ## Generate some bathymetry to test on

    bathy = np.random.random((100,100)) * (- 100)
    bathy = xr.DataArray(
                bathy, 
                dims=["lata","lona"],
                coords={"lata":np.linspace(yextent[0]-5,yextent[1]+5,100),
                        "lona":np.linspace(xextent[0]-5,xextent[1]+5,100)})
    # name the bathymetry variable of xarray dataarray
    bathy.name = "elevation"

    bathy.to_netcdf("bathy.nc",mode = "a")
    bathy.close()
    expt.bathymetry(
        'bathy.nc',
        {"xh":"lona",
        "yh":"lata",
        "elevation":"elevation"},
        minimum_layers = 1
    )

    ## Make an IC file to test on

    

    return


test_experiment(
    [-5,5],[0,10],
         ["2003-01-01 00:00:00","2003-01-01 00:00:00"],
         0.1,
         5,
         1,
         1000,
         "rundir/",
         "inputdir/",
         "toolpath",
         "even_spacing"
)
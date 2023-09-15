import numpy as np
import pytest
from regional_mom6 import experiment
import subprocess
import xarray as xr


# reload(experiment)
@pytest.mark.parametrize(
    (
        "xextent",
        "yextent",
        "daterange",
        "resolution",
        "vlayers",
        "dz_ratio",
        "depth",
        "mom_run_dir",
        "mom_input_dir",
        "toolpath",
        "gridtype",
    ),
    [
        (
            [-5, 5],
            [0, 10],
            ["2003-01-01 00:00:00", "2003-01-01 00:00:00"],
            0.1,
            5,
            1,
            1000,
            "rundir/",
            "inputdir/",
            "toolpath",
            "even_spacing",
        ),
    ],
)
def test_bathymetry(
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
    gridtype,
):
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
        gridtype,
    )

    ## Generate some bathymetry to test on

    bathy = np.random.random((100, 100)) * (-100)
    bathy = xr.DataArray(
        bathy,
        dims=["lata", "lona"],
        coords={
            "lata": np.linspace(yextent[0] - 5, yextent[1] + 5, 100),
            "lona": np.linspace(xextent[0] - 5, xextent[1] + 5, 100),
        },
    )
    # name the bathymetry variable of xarray dataarray
    bathy.name = "elevation"

    bathy.to_netcdf("bathy.nc", mode="a")
    bathy.close()

    # Now use this bathymetry as input in `expt.bathymetry()`
    expt.bathymetry(
        "bathy.nc",
        {"xh": "lona", "yh": "lata", "elevation": "elevation"},
        minimum_layers=1,
        chunks={"lat": 10, "lon": 10},
    )

    print(subprocess.run("rm bathy.nc", shell=True))
    ## Make an IC file to test on

    return


import numpy as np
import pytest
from regional_mom6 import experiment
import subprocess
import xarray as xr

# reload(experiment)


@pytest.mark.parametrize(
    (
        "xextent",
        "yextent",
        "daterange",
        "resolution",
        "vlayers",
        "dz_ratio",
        "depth",
        "mom_run_dir",
        "mom_input_dir",
        "toolpath",
        "gridtype",
    ),
    [
        (
            [-5, 5],
            [0, 10],
            ["2003-01-01 00:00:00", "2003-01-01 00:00:00"],
            0.1,
            5,
            1,
            1000,
            "rundir/",
            "inputdir/",
            "toolpath",
            "even_spacing",
        ),
    ],
)
def test_ocean_forcing(
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
    gridtype,
):
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
        gridtype,
    )

    ## Generate some bathymetry to test on

    initial_cond = xr.Dataset(
        {
            "temp": xr.DataArray(
                np.random.random((100, 100, 10)),
                dims=["lata", "lona", "deepness"],
                coords={
                    "lata": np.linspace(yextent[0] - 5, yextent[1] + 5, 100),
                    "lona": np.linspace(xextent[0] - 5, xextent[1] + 5, 100),
                    "deepness": np.linspace(0, 1000, 10),
                },
            ),
            "eta": xr.DataArray(
                np.random.random((100, 100)),
                dims=["lata", "lona"],
                coords={
                    "lata": np.linspace(yextent[0] - 5, yextent[1] + 5, 100),
                    "lona": np.linspace(xextent[0] - 5, xextent[1] + 5, 100),
                },
            ),
            "salt": xr.DataArray(
                np.random.random((100, 100, 10)),
                dims=["lata", "lona", "deepness"],
                coords={
                    "lata": np.linspace(yextent[0] - 5, yextent[1] + 5, 100),
                    "lona": np.linspace(xextent[0] - 5, xextent[1] + 5, 100),
                    "deepness": np.linspace(0, 1000, 10),
                },
            ),
            "u": xr.DataArray(
                np.random.random((100, 100, 10)),
                dims=["lata", "lona", "deepness"],
                coords={
                    "lata": np.linspace(yextent[0] - 5, yextent[1] + 5, 100),
                    "lona": np.linspace(xextent[0] - 5, xextent[1] + 5, 100),
                    "deepness": np.linspace(0, 1000, 10),
                },
            ),
            "v": xr.DataArray(
                np.random.random((100, 100, 10)),
                dims=["lata", "lona", "deepness"],
                coords={
                    "lata": np.linspace(yextent[0] - 5, yextent[1] + 5, 100),
                    "lona": np.linspace(xextent[0] - 5, xextent[1] + 5, 100),
                    "deepness": np.linspace(0, 1000, 10),
                },
            ),
        }
    )

    eastern_boundary = xr.Dataset(
        {
            "temp": xr.DataArray(
                np.random.random((100, 5, 10, 10)),
                dims=["lata", "lona", "deepness", "time"],
                coords={
                    "lata": np.linspace(yextent[0] - 5, yextent[1] + 5, 100),
                    "lona": np.linspace(xextent[1] - 0.5, xextent[1] + 0.5, 5),
                    "deepness": np.linspace(0, 1000, 10),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
            "eta": xr.DataArray(
                np.random.random((100, 5, 10)),
                dims=["lata", "lona", "time"],
                coords={
                    "lata": np.linspace(yextent[0] - 5, yextent[1] + 5, 100),
                    "lona": np.linspace(xextent[1] - 0.5, xextent[1] + 0.5, 5),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
            "salt": xr.DataArray(
                np.random.random((100, 5, 10, 10)),
                dims=["lata", "lona", "deepness", "time"],
                coords={
                    "lata": np.linspace(yextent[0] - 5, yextent[1] + 5, 100),
                    "lona": np.linspace(xextent[1] - 0.5, xextent[1] + 0.5, 5),
                    "deepness": np.linspace(0, 1000, 10),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
            "u": xr.DataArray(
                np.random.random((100, 5, 10, 10)),
                dims=["lata", "lona", "deepness", "time"],
                coords={
                    "lata": np.linspace(yextent[0] - 5, yextent[1] + 5, 100),
                    "lona": np.linspace(xextent[1] - 0.5, xextent[1] + 0.5, 5),
                    "deepness": np.linspace(0, 1000, 10),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
            "v": xr.DataArray(
                np.random.random((100, 5, 10, 10)),
                dims=["lata", "lona", "deepness", "time"],
                coords={
                    "lata": np.linspace(yextent[0] - 5, yextent[1] + 5, 100),
                    "lona": np.linspace(xextent[1] - 0.5, xextent[1] + 0.5, 5),
                    "deepness": np.linspace(0, 1000, 10),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
        }
    )

    subprocess.run("mkdir dummyinputs", shell=True)
    eastern_boundary.to_netcdf("dummyinputs/east_unprocessed", mode="a")
    initial_cond.to_netcdf("dummyinputs/ic_unprocessed", mode="a")
    eastern_boundary.close()
    initial_cond.close()

    expt.ocean_forcing(
        "dummyinputs",
        {
            "x": "lona",
            "y": "lata",
            "time": "time",
            "eta": "eta",
            "zl": "deepness",
            "u": "u",
            "v": "v",
            "tracers": {"temp": "temp", "salt": "salt"},
        },
        boundaries=["east"],
        gridtype="A",
    )

    return

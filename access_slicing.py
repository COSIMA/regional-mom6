import xarray as xr
import pandas as pd
import glob
from regional_library import nicer_slicer

'''
So, for access, we could write a script that simply picks out the ryf forcing for the year with daily 3d data, and does all of the necessary slicing, time rotating etc. 

Or we have more simple functions that we can call repetitively within the notebook - let's start with this one and then maybe automate it more efficiently. 
'''

def open_ryf_global(outputs=range(1077,1082)):
    filepaths = []
    for i in outputs:
        pattern = f'/g/data/ik11/outputs/access-om2-01/01deg_jra55v13_ryf9091/output{i}/ocean/ocean_daily*'
        matches = glob.glob(pattern)
        filepaths.extend(matches)
    om2_input = xr.open_mfdataset(filepaths, decode_times=True,
                                 parallel=True,
                                 chunks={'time':-1, 'yu_ocean':300, 'yt_ocean':300, 'xu_ocean':300, 'xt_ocean':300})
    return om2_input

def select_slice(d, borders, boundary = 'east'):
    yextent = borders[0]; xextent = borders[1]
    if boundary in ['east','west']:
        y1 = yextent[0]; y2 = yextent[1]
        if boundary == 'east':
            x1 = xextent[1]
            x2 = x1
        else:
            x1 = xextent[0]
            x2 = x1
    else:
        x1 = xextent[0]; x2 = xextent[0]
        if boundary == 'south':
            y1 = yextent[0]; y2 = y1
        else:
            y1 = yextent[1]; y2 = y1
    bound = d.sel(
        yu_ocean = slice(y1 - 0.2, y2 + 0.2),
        yt_ocean = slice(y1 - 0.2, y2 + 0.2),
        xu_ocean = slice(x1-0.2, x2+0.2),
        xt_ocean = slice(x1-0.2, x2+0.2))
    # bound = nicer_slicer(bound, [x1, x2], ["xu_ocean", "xt_ocean"])
    return bound

def time_rotate(d, run_year = 2170):
    before_start_time = f"{run_year}-12-31"
    after_end_time = f"{run_year+1}-01-01"

    left = d.sel(time=slice(after_end_time, None))
    left["time"] = pd.date_range("1991-01-01 12:00:00", periods=120)

    right = d.sel(time=slice(None, before_start_time))
    right["time"] = pd.date_range("1991-05-01 12:00:00", periods=245)

    return xr.concat([left, right], "time")
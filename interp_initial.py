import netCDF4
import xarray as xr
import xesmf as xe

dg = xr.open_dataset("/g/data/x77/ahg157/inputs/mom6/eac-01/hgrid_01.nc")[["x", "y"]]

# B-grid velocities cut from ocean_daily_3d_u/v:
#
# ncks -d time,30 -d yu_ocean,694,1184 -d xu_ocean,639,959 --fix_rec_dmn time -v u
#     /g/data/ik11/outputs/access-om2-01/01deg_jra55v13_ryf9091/output1077/ocean/ocean_daily_3d_u.nc init_u.nc
# ncks -d time,30 -d yu_ocean,694,1184 -d xu_ocean,639,959 --fix_rec_dmn time -v u
#     /g/data/ik11/outputs/access-om2-01/01deg_jra55v13_ryf9091/output1077/ocean/ocean_daily_3d_v.nc init_v.nc
# ncks -A init_u.nc init_v.nc && mv init_v.nc init_vel.nc
# ncwa -O -a time init_vel.nc init_vel.nc
# ncks -O -v time -x init_vel.nc init_vel.nc

di = xr.open_dataset("/g/data/x77/ahg157/inputs/mom6/eac-01/forcing/init_vel_bgrid.nc")

dg = dg.rename({"x": "lon", "y": "lat"})
di = di.rename({"xu_ocean": "lon", "yu_ocean": "lat"})

dg_u = dg.isel(nxp=slice(None, None, 2), nyp=slice(1, None, 2))
dg_v = dg.isel(nxp=slice(1, None, 2), nyp=slice(None, None, 2))

regridder_u = xe.Regridder(
    di, dg_u, "bilinear",
)
regridder_v = xe.Regridder(
    di, dg_v, "bilinear",
)

u_on_c = regridder_u(di.u)
u_on_c = u_on_c.rename({"lon": "xq", "lat": "yh", "nyp": "ny"})
u_on_c.name = "u"

v_on_c = regridder_v(di.v)
v_on_c = v_on_c.rename({"lon": "xh", "lat": "yq", "nxp": "nx"})
v_on_c.name = "v"

do = xr.merge((u_on_c, v_on_c))
do.st_ocean.attrs["axis"] = "Z"

do.to_netcdf(
    "/g/data/x77/ahg157/inputs/mom6/eac-01/forcing/init_vel_cgrid.nc",
    encoding={
        "u": {"_FillValue": netCDF4.default_fillvals["f4"]},
        "v": {"_FillValue": netCDF4.default_fillvals["f4"]},
    },
)

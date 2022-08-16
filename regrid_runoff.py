from pykdtree.kdtree import KDTree
import numpy as np
import xarray as xr

# ocean mask and supergrid (reduced to tracer points) for the target grid
dm = xr.open_dataset("ocean_mask.nc").rename({"nx": "longitude", "ny": "latitude"})
dg = (
    xr.open_dataset("hgrid_01.nc")
    .isel(nxp=slice(1, None, 2), nyp=slice(1, None, 2))
    .rename({"nyp": "latitude", "nxp": "longitude"})
)

# merge areas to get full cell area
area = dg.area
area["ny"] = area.ny // 2
area["nx"] = area.nx // 2
area = (
    area
    .stack(cell=["ny", "nx"])
    .groupby("cell")
    .sum()
    .unstack("cell")
)

# calculate coastal mask
cst = xr.zeros_like(dm.mask)
for dim in ["longitude", "latitude"]:
    for off in [-1, 1]:
        cst = xr.where((dm.mask > 0) & (dm.mask.shift(**{dim: off}) == 0), 1, cst)

# indices of coast points -- Nx2, first column are y indices, then x indices
cst_pts = np.vstack(np.nonzero(cst)).T
# coords of coast points -- Nx2, first column are latitudes, then longitudes
cst_coords = xr.concat((dg.y, dg.x + 360), "d").data.reshape(2, -1).T[np.flatnonzero(cst)]
cst_areas = area.data.flatten()[np.flatnonzero(cst)]

kd = KDTree(cst_coords)

# open the runoff section and construct its corner points
dr = xr.open_dataset("runoff_box.nc")
res = 0.25
lons = np.arange(dr.longitude[0] - res/2, dr.longitude[-1] + res, res)
lats = np.arange(dr.latitude[0] - res/2,  dr.latitude[-1] + res, res)

# source coords for remapping
runoff_coords = np.c_[np.meshgrid(dr.latitude, dr.longitude, indexing="ij")].reshape(2, -1).T
# coords for cell area calculation
corner_lat, corner_lon = np.meshgrid(np.deg2rad(lats), np.deg2rad(lons), indexing="ij")
Re = 6378.137e3
runoff_areas = np.abs(
    ((corner_lon[1:,1:] - corner_lon[:-1,:-1]) * Re**2) * (np.sin(corner_lat[1:,1:]) - np.sin(corner_lat[:-1,:-1]))
)

# nearest coastal point for every runoff point
_, nearest_cst = kd.query(runoff_coords)

# create output DataArray
runoff = xr.DataArray(
    0.0,
    {"time": dr.time, "latitude": dg.y.isel(longitude=0), "longitude": dg.x.isel(latitude=0)},
    ["time", "latitude", "longitude"]
)
runoff.name = "friver"
runoff.time.attrs["modulo"] = " "

ind_y = xr.DataArray(cst_pts[:,0], dims="coast")
ind_x = xr.DataArray(cst_pts[:,1], dims="coast")

for i in range(dr.time.size):
    # list of nearest coast point (on target grid), with the source data
    dat = np.c_[nearest_cst, (dr.friver[i].data * runoff_areas).flatten()]
    dat = dat[dat[:,0].argsort()] # sort by coast point idx

    # group by destination point
    cst_point, split_idx = np.unique(dat[:,0], return_index=True)
    cst_point = cst_point.astype(int)
    split_idx = split_idx[1:]

    # sum per destination point
    dat_cst = [x.sum() for x in np.split(dat[:,1], split_idx)]

    # assign the target value
    runoff[i, ind_y[cst_point], ind_x[cst_point]] = dat_cst / cst_areas[cst_point]

runoff.to_netcdf("runoff_regrid.nc", unlimited_dims="time")

import xarray as xr

print('Merging ERA5 data files...')
ds = xr.open_mfdataset('Data/Input_data/CDS/era5*.nc', combine='by_coords')
ds = ds.sel(latitude=slice(90, 0))
ds.to_netcdf('Data/Input_data/Z500_1999_2026_NH_ERA5.nc')
print('Done!')
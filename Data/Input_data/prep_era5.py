# prepares the merged dataset from multiple ERA5 data files, 
# subsetting to the Northern Hemisphere and converting geopotential to geopotential height

import xarray as xr

print('Merging ERA5 data files...')
ds = xr.open_mfdataset('CDS/era5*.nc', combine='by_coords')
ds = ds.chunk({'valid_time': 500})

print('Subsetting and adjusting coordinates...')
ds = ds.squeeze(['pressure_level'], drop=True)
ds = ds.sel(latitude=slice(90, 0))
ds = ds.rename({'valid_time': 'time'})

ds = ds.assign_coords(
    latitude=ds.latitude.astype('float32'),
    longitude=ds.longitude.astype('float32')
)

# shift from 0...360 to -180...180 and sort
ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
ds = ds.sortby('longitude')

g = 9.80665
print('Converting geopotential to geopotential height...')
ds['z'] = ds['z'] / g
ds['z'].attrs['units'] = 'm'
ds['z'].attrs['description'] = 'Geopotential height at 500 hPa'

print('Saving merged dataset to netCDF...')
ds.to_netcdf('Z500_1999_2026_NH_ERA5.nc', engine='h5netcdf')

print('Done!')
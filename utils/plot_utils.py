import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import os
import cartopy.crs as ccrs
from cartopy.feature import COASTLINE

def get_spatial_config(config_path):
    # prepare all configs from sample config file
    if os.path.exists("spatial_config.npz"):
        spatial_config = np.load("spatial_config.npz")
        lev = spatial_config["lev"]
        lat = spatial_config["lat"]
        lon = spatial_config["lon"]
    else:
        sample_nc = nc.Dataset(config_path)
        lev = sample_nc.variables["lev"][:]
        lat = sample_nc.variables["lat"][:]
        lon = sample_nc.variables["lon"][:]
        np.savez("spatial_config.npz", lev=lev, lat=lat, lon=lon)
    return lon, lat, lev

def plot_spatial(lon, lat, data, cbar_label=None, title=None, cmap="bwr",
                  vmin=None, vmax=None, norm=None, extend="both"):
    # Using PlateCarree projection which is commonly used for lat/lon data.
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))  
    if vmin is not None:
        img = ax.contourf(lon, lat, data, np.linspace(vmin, vmax, 20), cmap=cmap, 
                      transform=ccrs.PlateCarree(), norm=norm, extend=extend)
    else:
        img = ax.contourf(lon, lat, data, 20, cmap=cmap, 
                        transform=ccrs.PlateCarree(), norm=norm, extend=extend)
    # plt.imshow(r2, cmap="bwr", vmin=0, vmax=1)
    # Draw and label gridlines
    # Add coastlines
    ax.add_feature(COASTLINE, linewidth=0.5, edgecolor='black')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, 
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False  # Don't display top x-axis labels
    gl.right_labels = False  # Don't display right y-axis labels
    cbar = plt.colorbar(img,shrink=0.7)
    if cbar_label is not None:
        cbar.set_label(cbar_label)
    if title is not None:
        plt.title(title)
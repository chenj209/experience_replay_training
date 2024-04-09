import numpy as np
def compute_omega(u, v, pressure_levels, longitude, latitude):
    # Constants
    earth_radius = 6378137  # Radius of Earth in meters
    deg_to_rad = np.pi / 180  # Conversion factor from degrees to radians
    
    # Convert latitude and longitude to radians for derivative calculation
    lat_rad = latitude * deg_to_rad
    lon_rad = longitude * deg_to_rad
    
    # Calculate distances in meters for each grid point
    print("lon_rad gradient:", np.gradient(lon_rad).shape)
    print("cos(lat_rad):", np.cos(lat_rad).shape)
    dx = np.cos(lat_rad)[None, :, None] * earth_radius * np.gradient(lon_rad)[None, None, :]
    print("dx:", dx.shape)
    dy = earth_radius * np.gradient(lat_rad)[:, None]
    print("dy:", dy.shape)
    dp = np.gradient(pressure_levels, axis=0)
    print("dp:", dp.shape)
    
    # Compute central differences for u and v
    du_dx = np.gradient(u, axis=2) / dx
    dv_dy = np.gradient(v, axis=1) / dy
    print("du_dx:", du_dx.shape)
    print("dv_dy:", dv_dy.shape)
    
    # Initialize omega array with zeros
    omega = np.zeros((31,96,144))
    print("omega:", omega.shape)
    
    # Loop over pressure levels (except the surface)
    # surface being zero
    for k in range(len(pressure_levels)-1,0,-1):
        # Compute domega/dp using the continuity equation
        domega_dp = -(du_dx[k,:,:] + dv_dy[k,:,:])
        
        # Integrate domega/dp with respect to p to get omega
        omega[k,:,:] = omega[k+1,:,:] + domega_dp * dp[k]
    
    return omega

def get_pmid_from_x(ps, hyam, hybm):
    # ps: surface pressure, shape (96, 144)
    ps_level = ps[np.newaxis, :, :] * np.array(hybm)[:, np.newaxis, np.newaxis]
    ps_level += np.array(hyam)[:, np.newaxis, np.newaxis] * 100000
    return ps_level # 30x96x144 pmid

if __name__ == "__main__":
    from dataloader_utils import get_index_from_colnames
    import sys
    # Example data
    pconst = np.load("../consts/phys_consts.npz")
    col_names = np.loadtxt("../analysis/test_data/col_names.txt", dtype=str)
    hyam = pconst["hyam"]
    hybm = pconst["hybm"]
    sample_file = np.load("../analysis/test_data/08689.npy")
    print(sample_file.shape)
    ps_idx = get_index_from_colnames(col_names, "SPPS")
    print(ps_idx)
    ps = sample_file[ps_idx[0]:ps_idx[1]].squeeze()
    pmid = get_pmid_from_x(ps, hyam, hybm)
    u_idx = get_index_from_colnames(col_names, "UL")
    v_idx = get_index_from_colnames(col_names, "VL")
    u = sample_file[u_idx[0]:u_idx[1]]
    v = sample_file[v_idx[0]:v_idx[1]]
    print("u:", u.shape)
    print("v:", v.shape)
    print("pmid:", pmid.shape)
    # u = np.random.rand(10, 5, 5)  # Random u component
    # v = np.random.rand(10, 5, 5)  # Random v component
    # pressure_levels = np.linspace(1000, 100, 10)  # Pressure levels from 1000 hPa to 100 hPa
    # longitude = np.linspace(0, 360, 5)  # Longitude from 0 to 360 degrees
    # latitude = np.linspace(-90, 90, 5)  # Latitude from -90 to 90 degrees
    lon = np.linspace(0,357.5,144)
    lat = np.linspace(-90,90,96)

    omega = compute_omega(u, v, pmid, lon, lat)
    # print(omega)

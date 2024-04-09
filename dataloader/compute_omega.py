import numpy as np
def compute_omega(u, v, pressure_levels, longitude, latitude):
    # Constants
    earth_radius = 6371000  # Radius of Earth in meters
    deg_to_rad = np.pi / 180  # Conversion factor from degrees to radians
    
    # Convert latitude and longitude to radians for derivative calculation
    lat_rad = latitude * deg_to_rad
    lon_rad = longitude * deg_to_rad
    
    # Calculate distances in meters for each grid point
    dx = np.cos(lat_rad)[None, :, None] * earth_radius * np.gradient(lon_rad)
    dy = earth_radius * np.gradient(lat_rad)[:, None]
    dp = np.gradient(pressure_levels, axis=0)
    
    # Compute central differences for u and v
    du_dx = np.gradient(u, axis=2) / dx
    dv_dy = np.gradient(v, axis=1) / dy
    
    # Initialize omega array with zeros
    omega = np.zeros_like(u)
    
    # Loop over pressure levels (except the surface)
    for k in range(1, len(pressure_levels)):
        # Compute domega/dp using the continuity equation
        domega_dp = -(du_dx[:, k, :] + dv_dy[:, k, :])
        
        # Integrate domega/dp with respect to p to get omega
        omega[:, k, :] = omega[:, k-1, :] - domega_dp * dp[k]
    
    return omega

if __name__ == "__main__":
    # Example data
    u = np.random.rand(10, 5, 5)  # Random u component
    v = np.random.rand(10, 5, 5)  # Random v component
    pressure_levels = np.linspace(1000, 100, 10)  # Pressure levels from 1000 hPa to 100 hPa
    longitude = np.linspace(0, 360, 5)  # Longitude from 0 to 360 degrees
    latitude = np.linspace(-90, 90, 5)  # Latitude from -90 to 90 degrees

    omega = compute_omega(u, v, pressure_levels, longitude, latitude)
    print(omega)

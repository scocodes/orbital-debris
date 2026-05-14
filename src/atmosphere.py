import numpy as np
from pyatmos import download_sw_nrlmsise00, read_sw_nrlmsise00, nrlmsise00

class Atmosphere:
    "Initial Parameters"
    mu_earth = 3.986004418e14  # m^3/s^2   
    r_earth  = 6371e3   # m   
    rho0 = np.mean(np.array([15e-15, 20e-12]))    
    def __init__(self, lat=0.0, lon=0.0, date="2024-01-01 00:00:00"):
        self.lat = lat
        self.lon = lon
        self.date = date
        self.altitude_range = np.linspace(100, 1500, 290)

        self.swfile_nrl = download_sw_nrlmsise00()
        self.swdata_nrl = read_sw_nrlmsise00(self.swfile_nrl)

        "Atmospheric Density Parameters"
        self.rho_nrl = self.density_list()

    
    def density_list(self):
        rho_list = []
        for h in self.altitude_range:
            rho_nrlmsise00 = float(nrlmsise00(self.date, (self.lat, self.lon, h), self.swdata_nrl).rho)
            rho_list.append(rho_nrlmsise00)
            
        return np.array(rho_list)

    def density(self, alt_km):
        return np.interp(alt_km, self.altitude_range, self.rho_nrl)


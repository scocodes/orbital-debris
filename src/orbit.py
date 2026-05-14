import numpy as np
from src.atmosphere import Atmosphere

class Orbit:
    def __init__(self, height=850, B=1.227, manoeuvres=None, stop_alt_km=350, dt=10, atmosphere=None):
        self.height = height
        self.B = B
        self.manoeuvres = manoeuvres if manoeuvres is not None else []
        self.stop_alt_km = stop_alt_km
        self.dt = dt
        self.atmosphere = atmosphere
        
    "Initial conditions of the debris"
    def two_body_rhs(self, t, y):
        r = y[0:3]
        v = y[3:6]

        rmag = np.linalg.norm(r)
        alt_km = (rmag-Atmosphere.r_earth) / 1e3

        agrav = -Atmosphere.mu_earth*(r/rmag**3)
        rho = self.atmosphere.density(alt_km)
        vmag = np.linalg.norm(v)
        if vmag > 0:
            adrag = -1/(2*self.B) * (rho*v*vmag)
        else:
            adrag = np.zeros(3)
        
        atotal = adrag + agrav
        dydt = np.zeros_like(y)

        dydt[0:3] = v
        dydt[3:6] = atotal
        return dydt

    "RK4 Integrator"
    def rk4_step(self, fun, t, y, *args):
        k1 = fun(t, y, *args)
        k2 = fun(t+self.dt/2, y+self.dt/2*k1, *args)
        k3 = fun(t+self.dt/2, y+self.dt/2*k2, *args)
        k4 = fun(t+self.dt, y+self.dt*k3, *args)
        kavg = 1/6*(k1 + 2*k2 +2*k3 + k4)
        return y + self.dt*kavg

    "Main Orbtial Propagation"
    def orbitprop(self):
        
        height_km = self.height*1e3   
        r0_mag = Atmosphere.r_earth + height_km

        # Initial time parameters
        T = 2*np.pi*np.sqrt((r0_mag**3)/Atmosphere.mu_earth)
        t0 = 0
        tf = 100*T 

        # Initial Positional parameters
        r0 = np.array([r0_mag, 0.0, 0.0]) 
        vcirc = np.sqrt(Atmosphere.mu_earth/r0_mag)
        v0 = np.array([0.0, vcirc, 0.0])
        y0 = np.hstack((r0, v0))

        # Manoeuvres Call
        manoeuvres = sorted(self.manoeuvres, key=lambda m: m[0])
        man_index = 0

        ts = []
        ys = []

        t = t0
        y = y0.copy()

        ts.append(t)
        ys.append(y.copy())

        i = 0


        while True:
            i += 1

            while man_index < len(manoeuvres) and t >= manoeuvres[man_index][0]:
                _, dv = manoeuvres[man_index]
                y[3:6] += dv
                man_index += 1

            y = self.rk4_step(self.two_body_rhs, t, y)
            t = t + self.dt

            if t >= 3600*24*100:
                return np.array(ts), np.array(ys)
            else:     
                ts.append(t)
                ys.append(y.copy())

            if self.stop_alt_km is not None:
                rmag = np.linalg.norm(y[0:3])
                alt_km = (rmag - Atmosphere.r_earth) / 1e3
                if alt_km <= self.stop_alt_km:
                    self.ts = np.array(ts[:i+1])
                    self.ys = np.array(ys[:i+1])
                    return self.ts, self.ys
                
    "Altitude Conversion Function"
    def althist(self, ys):
        rvec = ys[:, 0:3]
        radmag = np.linalg.norm(rvec, axis=1)
        self.altkm = (radmag - Atmosphere.r_earth) / 1e3
        return self.altkm

    "Re-Entry Height Condition Check"
    def validity(self, ys):
        altitude_valid = self.althist(ys)
        return np.any(altitude_valid <= 350)
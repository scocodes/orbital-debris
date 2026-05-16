import numpy as np
import pandas as pd

class Data:
    def __init__(self, atmosphere, orbit):
        self.atmosphere = atmosphere
        self.orbit = orbit

    def log_uniform(self, low, high):
        return 10**np.random.uniform(np.log10(low), np.log10(high))
    
    def ballistic(self, area, mass):
        cd = 2.2
        return mass/(cd*area)
    
    def data_collection(self, trial_number):
        output_X = []
        parameters_physical = ["Mass", "Area", "Altitude", 
                               "Impulse Magnitude", "Number of Burns", "Number of Burns Uncertainty",
                               "Impulse Direction Uncertainty", "Impulse Magnitude Uncertainty",
                               "Ballistic Coefficient", "Impulse Application Uncertainty", "Re-Entry Achieved?"]

        for n in range(trial_number):
            

            mass_trial = self.log_uniform(0.01, 1000)
            area_trial = self.log_uniform(0.01, 1000)
            altitude_trial = self.log_uniform(350, 1500)
            impulse_magnitude_trial = self.log_uniform(30, 300)
            n_burns_trial = np.random.randint(1, 21)
            n_burns_uncertainty = np.random.normal(1, 0.3)
            impulse_direction_uncertainty = np.random.normal(0, 18)
            impulse_magnitude_uncertainty = np.random.normal(0, 0.3)
            B_trial = self.ballistic(area_trial, mass_trial)

            
            "Modify previous functions to allow inputs of trial parameters"

            
            deltavtrial = impulse_magnitude_trial * impulse_magnitude_uncertainty
            deltavtrial = max(deltavtrial, 0)
            r0_mag = self.atmosphere.r_earth + altitude_trial * 1e3
            T = 2*np.pi*np.sqrt((r0_mag**3)/self.atmosphere.mu_earth)
            impulse_application_uncertainy = np.random.normal(0, (0.3*T))

            theta = np.deg2rad(impulse_direction_uncertainty)
            u_rot = np.array([-np.sin(theta), np.cos(theta), 0])
            dv_vec = -deltavtrial * u_rot
        
            manoeuvres = []

            n_burns_trial = n_burns_trial * np.abs(n_burns_uncertainty)

            for i in range(int(n_burns_trial)):
                u_time = impulse_application_uncertainy
                tburn = i*T + u_time
                manoeuvres.append((tburn, dv_vec))

            orbit_trial = self.orbit(altitude_trial, B_trial, manoeuvres, stop_alt_km=350, dt=10, atmosphere=self.atmosphere)
            _, ys = orbit_trial.orbitprop()
            valid = orbit_trial.validity(ys)

            if valid == False:
                valid_label = "No"
            else:
                valid_label = "Yes"

            output_X.append([round(mass_trial, 3), round(area_trial, 3), round(altitude_trial, 3), round(impulse_magnitude_trial, 3),
                             round(n_burns_trial, 3), round(n_burns_uncertainty, 3), round(impulse_direction_uncertainty, 3), 
                             round(impulse_magnitude_uncertainty, 3), round(B_trial, 3), round(impulse_application_uncertainy, 3), valid_label])

        df = pd.DataFrame(output_X, columns=parameters_physical)
        df.to_csv("data/dataset_test.csv", index=False)
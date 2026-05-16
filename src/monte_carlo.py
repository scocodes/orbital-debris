import numpy as np
import matplotlib.pyplot as plt 

class MonteCarlo:
    def __init__(self, N_trials, deltav, n_burns, height, B, orbit=None, atmosphere=None):
        self.N_trials = N_trials
        self.deltav = deltav
        self.n_burns = n_burns
        self.height = height
        self.B = B
        self.orbit = orbit
        self.atm = atmosphere
    
    "MonteCarlo Uncertainty Engine"
    def montecarlo(self):
        results = []

        for trial in range(self.N_trials):
            uncert = 1 + np.random.normal(0, 0.2)
            deltavtrial = self.deltav * uncert

            r0_mag = self.atmosphere.r_earth + self.height * 1e3
            theta = np.deg2rad(np.random.normal(0, 18))
            u_rot = np.array([-np.sin(theta), np.cos(theta), 0])
            dv_vec = -deltavtrial * u_rot
        
            manoeuvres = []

            T = 2*np.pi*np.sqrt((r0_mag**3)/self.tmosphere.mu_earth)
            
            for i in range(self.n_burns):
                u_time = np.random.normal(0, 1*(0.2*T))
                tburn = i*T + u_time
                manoeuvres.append((tburn, dv_vec))

            _, ys = self.orbit.orbitprop(self.height, self.B, manoeuvres, stop_alt_km=350)
            valid = self.orbit.validity(ys)

            results.append((trial, dv_vec, valid))

        return results

    "Single Case Decay Time Function"
    def orbitPropCheck(self):
        ts, ys = self.orbit.orbitprop(manoeuvres=None, stop_alt_km=100)
        altitude_00 = self.orbit.althist(ys)
        plt.plot(ts/(24*3600), altitude_00)
        plt.xlabel("Time (Days)")
        plt.ylabel("Altitude (km)")
        plt.grid(True)
        plt.show()



    "Optimal use case with Uncertainty Function"
    def optiMc(self, deltav_range):
        results = []
        for dv in deltav_range:
            for nb in self.n_burns:
                mc = Orbit.montecarlo(self)
                p_success = np.mean([ok for _, _, ok in mc])
                results.append((self.N_trials, dv, nb, p_success))
        results = sorted(results, key=lambda x: (-x[3], x[1], x[2])) 

        df = pd.DataFrame(results, columns=[
        "N_trials",
        "Delta V (m/s)",
        "N_burns",
        "p_success"
        ])

        df.to_excel("test 1.xlsx", index=False)         
        return results


    "Error Convergence Plotting Function"
    def error_plotter(self):
        # Initial Parameters
        target = 0.01

        #Iteration Start Values
        n = 100
        k = 0
        n_trials = n
        p_success = [0.0]
        n_list = []
        error_list = []
        total_success = 0

        while True:

            k += 1

            mc = self.montecarlo(self)
            prob_trial = np.mean([ok for _, _, ok in mc])

            batch_successes = np.sum([ok for _, _, ok in mc])
            total_success += batch_successes
            
            print(f"Trial number {n_trials} complete")
            
            if k == 1:
                recurrance = prob_trial
            else:
                recurrance = (((k-1)*p_success[-1])+ prob_trial)/k
            
            p_success.append(recurrance)
            error = p_success[-1]-p_success[-2]

            if p_success[-2] > 0:
                error_percentage = np.abs((error/p_success[-1])) * 100
            else:
                error_percentage = np.abs(error*100)
            print(f"Error Percentage: {error_percentage}")

            error_list.append(error_percentage)
            n_list.append(n_trials)
            

            if n_trials >= 3000:
                p_hat = total_success/n_trials
                print(f"P hat = {p_hat}")
                standard_error = np.sqrt(p_hat*(1-p_hat)/n_trials)
                print(f"Standard Error: {standard_error}")
                break
            
            elif np.abs(error_percentage) <= target:
                p_hat = total_success/n_trials
                print(f"P hat = {p_hat}")
                standard_error = np.sqrt(p_hat*(1-p_hat)/n_trials)
                print(f"Standard Error: {standard_error}")
                break

            #Updates
            n_trials += n
    
        plt.plot(n_list, error_list, label="Convergence Error in Success Probability")
        plt.xlabel("Number of Trials")
        plt.ylabel("Error (%)")
        plt.legend()
        plt.grid(True)
        plt.show()
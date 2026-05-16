from src.atmosphere import Atmosphere
from src.orbit import Orbit
from src.monte_carlo import MonteCarlo
from src.data_collection import Data
from src.machine_learning import MachineLearning


def main():
    atm = Atmosphere()
    orb = Orbit()
    data = Data(atm, orb)
    ml = MachineLearning(data)
    ml_single = ml.prediction(1, 1, 1000, 90, 20, 0.2, 0.2, 0.2, 0.2)
    
    return ml_single

main()
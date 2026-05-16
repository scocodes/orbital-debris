Orbital Debris ML Surrogate

Overview

This project develops a machine learning surrogate model for predicting orbital debris re-entry outcomes following laser-induced impulse perturbations.

The repository combines:

* orbital propagation
* atmospheric drag modelling
* Monte Carlo uncertainty analysis
* machine learning classification
* surrogate modelling techniques

to rapidly predict orbital decay success without requiring full numerical propagation for every case.

The project was developed as an extension of stochastic orbital debris removal research investigating laser-based active debris removal (ADR) in Low Earth Orbit (LEO).

⸻

Features

* Two-body orbital propagation
* Atmospheric drag modelling using NRLMSISE-00
* RK4 numerical integration
* Monte Carlo uncertainty simulation
* Laser-induced impulsive manoeuvre modelling
* Synthetic dataset generation
* Machine learning surrogate modelling
* Classification of re-entry success
* Multiple ML model comparison
* Feature importance analysis
* Fast inference prediction system

Repository Structure

orbital-debris-ml/
│
├── main.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── dataset_test.csv
│
└── src/
    ├── atmosphere.py
    ├── orbit.py
    ├── monte_carlo.py
    ├── data_collection.py
    ├── machine_learning.py

Methodology

1. Orbital Propagation

The orbital dynamics model propagates debris trajectories using:

* two-body gravitational dynamics
* atmospheric drag
* impulsive velocity perturbations

State propagation is performed using a fourth-order Runge-Kutta (RK4) numerical integrator.

⸻

2. Atmospheric Modelling

Atmospheric density is modelled using the NRLMSISE-00 atmospheric model via the pyatmos package.

Density interpolation is used during orbital propagation to calculate drag acceleration as a function of altitude.

⸻

3. Monte Carlo Simulation

Monte Carlo analysis is used to model uncertainty in:

* impulse magnitude
* impulse direction
* impulse application timing
* number of burns

Thousands of stochastic trajectories are generated to evaluate re-entry probability.

⸻

4. Machine Learning Surrogate Modelling

Synthetic datasets generated from orbital simulations are used to train surrogate machine learning models capable of rapidly predicting orbital decay success.

The following models are implemented and compared:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* XGBoost Classifier

⸻

Machine Learning Features

Input features include:

* mass
* area
* ballistic coefficient
* initial altitude
* impulse magnitude
* number of burns
* impulse uncertainties
* burn timing uncertainty

Output target:

* re-entry achieved (Yes/No)

⸻

Results

Initial testing demonstrated:

* Logistic Regression achieved lower accuracy due to the nonlinear nature of orbital decay dynamics.
* Tree-based ensemble models significantly improved predictive performance.
* Random Forest and XGBoost provided the strongest classification performance.

The results suggest that orbital decay success exhibits highly nonlinear interactions between altitude, drag effects, ballistic coefficient and applied impulse characteristics.
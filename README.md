# Orbital Debris ML Surrogate Model

## Overview

This project investigates machine-learning-based surrogate modelling for stochastic orbital debris propagation under laser-induced impulsive manoeuvres.

A high-fidelity orbital propagator incorporating:
- two-body orbital dynamics
- atmospheric drag
- RK4 numerical integration
- NRLMSISE-00 atmospheric density modelling
- Monte Carlo uncertainty propagation

is used to generate datasets for surrogate machine learning models capable of rapidly approximating orbital decay outcomes.

The project is an extension of ongoing research into laser-induced active debris removal (ADR).

---

## Features

- RK4 orbital propagation
- Atmospheric drag modelling using NRLMSISE-00
- Monte Carlo uncertainty analysis
- Laser-induced impulsive manoeuvre modelling
- Automated dataset generation
- ML-ready surrogate modelling pipeline

---

## Repository Structure

```orbital-debris-ml
src/
    __init__.py
    atmosphere.py
    data_collection.py
    monte_carlo.py
    orbit.py   
data/

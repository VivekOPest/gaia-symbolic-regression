"""
=========================================================
Synthetic Test using REAL Gaia Features
=========================================================

Purpose
-------
Uses the REAL Gaia temperatures, radii and distances.

Test 1
-------
Constructs an artificial flux

    F = T^4 * R^2 / D^2

and checks whether regression recovers

    T = 4
    R = 2
    D = -2

Test 2
-------
Repeats the regression using the actual Gaia flux_proxy.

Author
------
Vivek Choudhary
=========================================================
"""

import numpy as np
import pandas as pd

from scipy.optimize import least_squares


# -------------------------------------------------------
# Load Gaia dataset
# -------------------------------------------------------

DATASET = "data/gaia_features.csv"

df = pd.read_csv(DATASET)

T = df["temperature_K"].to_numpy()
R = df["radius_m"].to_numpy()
D = df["distance_m"].to_numpy()

# -------------------------------------------------------
# Remove invalid values
# -------------------------------------------------------

mask = (
    (T > 0)
    &
    (R > 0)
    &
    (D > 0)
)

df = df.loc[mask]

T = df["temperature_K"].to_numpy()
R = df["radius_m"].to_numpy()
D = df["distance_m"].to_numpy()

# =======================================================
# TEST 1
# PERFECT STEFAN-BOLTZMANN FLUX
# =======================================================

print("=" * 60)
print("TEST 1 : PERFECT PHYSICS")
print("=" * 60)

F = (T**4) * (R**2) / (D**2)

logT = np.log10(T)
logR = np.log10(R)
logD = np.log10(D)
logF = np.log10(F)


def residual(params):

    a, b, c, d = params

    prediction = (
        a * logT
        + b * logR
        + c * logD
        + d
    )

    return prediction - logF


initial_guess = [4, 2, -2, 0]

result = least_squares(
    residual,
    initial_guess,
    method="trf"
)

a, b, c, d = result.x

print("\nRecovered Exponents")
print("-------------------------")
print(f"T : {a:.10f}")
print(f"R : {b:.10f}")
print(f"D : {c:.10f}")
print(f"Intercept : {d:.10f}")

print("\nExpected")
print("-------------------------")
print("T : 4")
print("R : 2")
print("D : -2")

print("\nDifference")
print("-------------------------")
print(f"T : {a-4:.3e}")
print(f"R : {b-2:.3e}")
print(f"D : {c+2:.3e}")


# =======================================================
# TEST 2
# ACTUAL GAIA FLUX
# =======================================================

print("\n")
print("=" * 60)
print("TEST 2 : REAL GAIA FLUX")
print("=" * 60)

F = df["flux_proxy"].to_numpy()

mask = F > 0

logT = np.log10(T[mask])
logR = np.log10(R[mask])
logD = np.log10(D[mask])
logF = np.log10(F[mask])


def residual_real(params):

    a, b, c, d = params

    prediction = (
        a * logT
        + b * logR
        + c * logD
        + d
    )

    return prediction - logF


result = least_squares(
    residual_real,
    initial_guess,
    method="trf"
)

a, b, c, d = result.x

print("\nRecovered Exponents")
print("-------------------------")
print(f"T : {a:.10f}")
print(f"R : {b:.10f}")
print(f"D : {c:.10f}")
print(f"Intercept : {d:.10f}")

print("\nComparison")
print("-------------------------")
print("Perfect Physics")
print("T = 4")
print("R = 2")
print("D = -2")

print("\nGaia Flux")
print(f"T = {a:.6f}")
print(f"R = {b:.6f}")
print(f"D = {c:.6f}")
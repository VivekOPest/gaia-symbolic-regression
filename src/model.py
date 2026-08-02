"""
model.py

Neural Network for predicting Gaia Flux Proxy
from stellar parameters.

Inputs:
    - Temperature (K)
    - Radius (m)
    - Distance (m)
    - Physics Predictor (R²T⁴/d²)

Output:
    - Gaia Flux Proxy

Author:
    Vivek Choudhary

Project:
    Gaia Symbolic Regression
"""

import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# --------------------------------------------------
# Project Directories
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "gaia_features.csv"

RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = RESULTS_DIR / "model.pt"

HISTORY_PATH = RESULTS_DIR / "history.csv"

PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"

SCALER_PATH = RESULTS_DIR / "scaler.pkl"

RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Training Configuration
# --------------------------------------------------

RANDOM_STATE = 42

TEST_SIZE = 0.20

BATCH_SIZE = 256

LEARNING_RATE = 1e-3

MAX_EPOCHS = 300

EARLY_STOPPING_PATIENCE = 20


DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


print("=" * 60)
print("Gaia Neural Network Training")
print("=" * 60)

print(f"Project Root : {PROJECT_ROOT}")
print(f"Data File    : {DATA_PATH}")
print(f"Results      : {RESULTS_DIR}")
print(f"Device        : {DEVICE}")

print("=" * 60)



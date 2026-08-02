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

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully.\n")

print("\nShape:", df.shape)


# --------------------------------------------------
# Features
# --------------------------------------------------

X = df[
    [
        "temperature_K",
        "radius_m",
        "distance_m"
    ]
]

# Learn logarithm of flux instead of flux

y = np.log10(df["flux_proxy"])


# --------------------------------------------------
# Train Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.50,

    random_state=42,

    shuffle=True

)

print("\nTraining Samples :", len(X_train))

print("Testing Samples  :", len(X_test))

# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

joblib.dump(

    scaler,

    SCALER_PATH

)

# --------------------------------------------------
# PyTorch Dataset
# --------------------------------------------------

class GaiaDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y.values,
            dtype=torch.float32
        ).view(-1, 1)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]



train_dataset = GaiaDataset(

    X_train,

    y_train

)

test_dataset = GaiaDataset(

    X_test,

    y_test

)




train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False

)


print("\n")

print("Training batches :", len(train_loader))

print("Testing batches  :", len(test_loader))


# ==========================================================
# Neural Network Architecture
# ==========================================================

class GaiaNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(3, 64),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Dropout(0.20),


            nn.Linear(64, 64),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Dropout(0.20),


            nn.Linear(64, 32),

            nn.ReLU(),


            nn.Linear(32, 16),

            nn.ReLU(),


            nn.Linear(16, 1)

        )

    def forward(self, x):

        return self.network(x)


# ==========================================================
# Build Model
# ==========================================================

model = GaiaNet().to(DEVICE)

# ==========================================================
# Number of Trainable Parameters
# ==========================================================

total_parameters = sum(

    p.numel()

    for p in model.parameters()

    if p.requires_grad

)

print(f"\nTrainable Parameters : {total_parameters:,}")     

# ==========================================================
# Forward Pass Test
# ==========================================================

sample_batch = next(iter(train_loader))

X_batch, y_batch = sample_batch

X_batch = X_batch.to(DEVICE)

prediction = model(X_batch)

print("\nInput Shape :", X_batch.shape)

print("Prediction Shape :", prediction.shape)


print("\nModel Successfully Built.\n")
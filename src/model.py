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
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

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

TEST_SIZE = 0.50

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

# ==========================================================
# Loss Function
# ==========================================================

criterion = nn.MSELoss()

print("Loss Function :", criterion)

# ==========================================================
# Optimizer
# ==========================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4

)

print("Optimizer : AdamW")

# ==========================================================
# Learning Rate Scheduler
# ==========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=5

)

print("Scheduler : ReduceLROnPlateau")

# ==========================================================
# Training Variables
# ==========================================================

best_loss = np.inf

best_epoch = 0

patience_counter = 0

history = {

    "epoch": [],

    "train_loss": [],

    "test_loss": [],

    "learning_rate": []

}

# ==========================================================
# Training Loop
# ==========================================================

print("\nStarting Training...\n")

start_time = time.time()

for epoch in range(MAX_EPOCHS):

    # ---------------------------------------------
    # Training
    # ---------------------------------------------

    model.train()

    train_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)

        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        prediction = model(X_batch)

        loss = criterion(prediction, y_batch)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_loader.dataset)


    # ---------------------------------------------
    # Validation
    # ---------------------------------------------

    model.eval()

    test_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(DEVICE)

            y_batch = y_batch.to(DEVICE)

            prediction = model(X_batch)

            loss = criterion(prediction, y_batch)

            test_loss += loss.item() * X_batch.size(0)

    test_loss /= len(test_loader.dataset)


    # ---------------------------------------------
    # Scheduler
    # ---------------------------------------------

    scheduler.step(test_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # ---------------------------------------------
    # Save History
    # ---------------------------------------------

    history["epoch"].append(epoch + 1)

    history["train_loss"].append(train_loss)

    history["test_loss"].append(test_loss)

    history["learning_rate"].append(current_lr)


    # ---------------------------------------------
    # Save Best Model
    # ---------------------------------------------

    if test_loss < best_loss:

        best_loss = test_loss

        best_epoch = epoch + 1

        patience_counter = 0

        torch.save(

            {

                "epoch": best_epoch,

                "validation_loss": best_loss,

                "model_state_dict": model.state_dict(),

                "optimizer_state_dict": optimizer.state_dict()

            },

            MODEL_PATH

        )

    else:

        patience_counter += 1


    # ---------------------------------------------
    # Progress
    # ---------------------------------------------

    print(
        f"Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
        f"Train {train_loss:.6f} | "
        f"Test {test_loss:.6f} | "
        f"LR {current_lr:.2e}"
    )


    # ---------------------------------------------
    # Early Stopping
    # ---------------------------------------------

    if patience_counter >= EARLY_STOPPING_PATIENCE:

        print("\nEarly stopping triggered.\n")

        break

training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.1f} seconds.")


# ==========================================================
# Save Training History
# ==========================================================

history_df = pd.DataFrame(history)

history_df["elapsed_seconds"] = training_time

history_df.to_csv(

    HISTORY_PATH,

    index=False

)

print("\nTraining history saved.")

# ==========================================================
# Load Best Model
# ==========================================================

checkpoint = torch.load(

    MODEL_PATH,

    map_location=DEVICE

)

model.load_state_dict(

    checkpoint["model_state_dict"]

)

model.eval()

print("Best model loaded.")

print("\n" + "=" * 60)

print("Training Summary")

print("=" * 60)

print(f"Best Epoch        : {checkpoint['epoch']}")

print(f"Validation Loss   : {checkpoint['validation_loss']:.6f}")

print(f"Training Time     : {training_time:.2f} seconds")

print("=" * 60)


# ==========================================================
# Predictions
# ==========================================================

predictions = []

truth = []

model.eval()

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(DEVICE)

        output = model(X_batch)

        predictions.extend(

            output.cpu().numpy().flatten()

        )

        truth.extend(

            y_batch.numpy().flatten()

        )

predictions = 10 ** np.array(predictions)

truth = 10 ** np.array(truth)

# ----------------------------------------
# Prediction Errors
# ----------------------------------------

residuals = predictions - truth

percentage_error = (
    residuals / truth
) * 100

prediction_df = pd.DataFrame(

    {

        "True Flux": truth,

        "Predicted Flux": predictions,

        "Residual": residuals,

        "Percentage Error": percentage_error

    }

)

prediction_df.to_csv(

    PREDICTIONS_PATH,

    index=False

)

print("\nPredictions saved.")

rmse = np.sqrt(

    mean_squared_error(

        truth,

        predictions

    )

)

mae = mean_absolute_error(

    truth,

    predictions

)

r2 = r2_score(

    truth,

    predictions

)

mape = np.mean(

    np.abs(

        percentage_error

    )

)


print("\n")

print("=" * 60)

print("Model Performance")

print("=" * 60)

print(f"RMSE : {rmse:.3f}")

print(f"MAE  : {mae:.3f}")

print(f"MAPE : {mape:.3f}%")

print(f"R²   : {r2:.5f}")

print("=" * 60)

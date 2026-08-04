"""
model.py

Neural Network for predicting Gaia Flux Proxy
from stellar parameters.

Inputs:
    - Temperature (K)
    - Radius (m)
    - Distance (m)

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
import matplotlib.pyplot as plt


# --------------------------------------------------
# Project Directories
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "gaia_features.csv"

RESULTS_DIR = PROJECT_ROOT / "results"

from pathlib import Path
import re

RESULTS_DIR.mkdir(exist_ok=True)

existing = []

for folder in RESULTS_DIR.iterdir():

    if folder.is_dir():

        m = re.match(r"trial_ml(\d+)", folder.name)

        if m:

            existing.append(int(m.group(1)))

trial_number = max(existing, default=0) + 1

TRIAL_DIR = RESULTS_DIR / f"trial_ml{trial_number:03d}"

TRIAL_DIR.mkdir()

PLOT_DIR = TRIAL_DIR / "plots"
TABLE_DIR = TRIAL_DIR / "tables"

PLOT_DIR.mkdir()
TABLE_DIR.mkdir()


MODEL_PATH = TRIAL_DIR / "model.pt"

HISTORY_PATH = TRIAL_DIR / "history.csv"

PREDICTIONS_PATH = TRIAL_DIR / "predictions.csv"

SUMMARY_PATH = TRIAL_DIR / "summary.txt"

METRICS_PATH = TRIAL_DIR / "metrics.json"

CONFIG_PATH = TRIAL_DIR / "config.json"

SCALER_PATH = TRIAL_DIR / "scaler.pkl"

RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Training Configuration
# --------------------------------------------------

RANDOM_STATE = 42

TEST_SIZE = 0.50

BATCH_SIZE = 256

LEARNING_RATE = 2e-4

MAX_EPOCHS = 800

EARLY_STOPPING_PATIENCE = 370


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

X = pd.DataFrame({
    "temperature" : np.log10(df["temperature_K"]),
    "radius"      : np.log10(df["radius_m"]),
    "distance"    : np.log10(df["distance_m"]),
})

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

            nn.Linear(3, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.05),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.05),

            nn.Linear(128,64),
            nn.GELU(),

            nn.Linear(64,32),
            nn.GELU(),

            nn.Linear(32,1)
        )

    def forward(self, x):

        return self.network(x)


# ==========================================================
# Build Model
# ==========================================================

model = GaiaNet().to(DEVICE)

for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

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

criterion = nn.HuberLoss(delta=0.5)

print("Loss Function :", criterion)

# ==========================================================
# Optimizer
# ==========================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=5e-5

)

print("Optimizer : AdamW")

# ==========================================================
# Learning Rate Scheduler
# ==========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=40

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

#===========================================================
# Training Loop
# ==========================================================

print("\nStarting Training...\n")

config = {

    "epochs": MAX_EPOCHS,

    "batch_size": BATCH_SIZE,

    "learning_rate": LEARNING_RATE,

    "optimizer": "AdamW",

    "scheduler": "ReduceLROnPlateau",

    "loss": "HuberLoss",

    "architecture": [256,256,128,64,32,1],

    "dropout":[0.10,0.08,0.05],

    "device": str(DEVICE),

    "training_samples": len(train_dataset),

    "testing_samples": len(test_dataset),

    "parameters": total_parameters

}

import json

with open(CONFIG_PATH,"w") as f:

    json.dump(config,f,indent=4)

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

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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

predictions_log = []
truth_log = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(DEVICE)

        output = model(X_batch)

        predictions_log.extend(output.cpu().numpy().flatten())
        truth_log.extend(y_batch.numpy().flatten())

predictions_log = np.array(predictions_log)

truth_log = np.array(truth_log)


predictions = predictions_log.copy()
truth = truth_log.copy()

residuals = predictions - truth

rmse = np.sqrt(mean_squared_error(truth,predictions))

mae = mean_absolute_error(truth,predictions)

r2 = r2_score(truth,predictions)

mean_abs_log_error = np.mean(np.abs(residuals))

metrics = {

    # Dataset
    "Training Samples": len(train_dataset),
    "Testing Samples": len(test_dataset),

    # Training
    "Training Time (s)": float(training_time),
    "Best Epoch": int(best_epoch),
    "Validation Loss": float(best_loss),
    "Final Train Loss": float(history["train_loss"][-1]),
    "Final Validation Loss": float(history["test_loss"][-1]),

    # Prediction Accuracy
    "RMSE": float(rmse),
    "MAE": float(mae),
    "R²": float(r2),

    # Residual statistics
    "Residual Mean": float(np.mean(residuals)),
    "Residual Std": float(np.std(residuals)),
    "Residual Median": float(np.median(residuals)),
    "Residual Max": float(np.max(residuals)),
    "Residual Min": float(np.min(residuals)),

    # Prediction statistics
    "Prediction Mean": float(np.mean(predictions)),
    "Prediction Std": float(np.std(predictions)),

    # Truth statistics
    "Truth Mean": float(np.mean(truth)),
    "Truth Std": float(np.std(truth)),

    # Model
    "Parameters": int(total_parameters),
    "Learning Rate": LEARNING_RATE,
    "Batch Size": BATCH_SIZE
}

with open(METRICS_PATH,"w") as f:

    json.dump(metrics,f,indent=4)

with open(SUMMARY_PATH,"w") as f:

    f.write("GAIA Neural Network Trial\n")

    f.write("="*50+"\n")

    f.write(f"Trial : trial_ml{trial_number:03d}\n")

    f.write(f"Training samples : {len(train_dataset)}\n")

    f.write(f"Testing samples : {len(test_dataset)}\n")

    f.write(f"Best epoch : {best_epoch}\n")

    f.write(f"Validation loss : {best_loss:.6f}\n")

    f.write(f"RMSE (log10 Flux): {rmse:.4f}\n")

    f.write(f"MAE (log10 Flux): {mae:.4f}\n")

    f.write(f"Mean Absolute Log Error : {mean_abs_log_error:.4f}\n")

    f.write(f"R2 : {r2:.6f}\n")

    f.write(f"Training time : {training_time:.2f} sec\n")

plt.figure(figsize=(10,6))

plt.plot(
    history["epoch"],
    history["train_loss"],
    linewidth=2,
    label="Training Loss"
)

plt.plot(
    history["epoch"],
    history["test_loss"],
    linewidth=2,
    label="Validation Loss"
)

plt.title("Training History", fontsize=16)

plt.xlabel("Epoch")

plt.ylabel("Huber Loss")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOT_DIR/"loss_curve.png",
    dpi=300
)

plt.close()


plt.figure(figsize=(10,6))

plt.hist(
    residuals,
    bins=80
)

plt.title("Residual Distribution")

plt.xlabel("Residual (log10 Flux)")

plt.ylabel("Count")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOT_DIR/"residual_distribution.png",
    dpi=300
)

plt.close()


plt.figure(figsize=(10,6))

plt.scatter(
    truth,
    residuals,
    s=8,
    alpha=0.4
)

plt.axhline(
    0,
    color="red",
    linestyle="--"
)

plt.title("Residual vs True Flux")

plt.xlabel("True log10 Flux")
plt.ylabel("Residual (log10 Flux)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOT_DIR/"residual_vs_true.png",
    dpi=300
)

plt.close()

plt.figure(figsize=(8,8))

plt.scatter(
    truth,
    predictions,
    s=5,
    alpha=0.4
)

plt.plot(
    [truth.min(),truth.max()],
    [truth.min(),truth.max()],
    "r--"
)

plt.xlabel("True log10 Flux")
plt.ylabel("Predicted log10 Flux")
plt.title("Predicted vs True (log10 Flux)")

plt.grid(True)

plt.tight_layout()

plt.savefig(PLOT_DIR/"prediction_vs_truth.png",dpi=300)

plt.close()



prediction_df = pd.DataFrame({

    "True log10 Flux": truth_log,
    "Predicted log10 Flux": predictions_log,

    "True Flux": 10**truth_log,
    "Predicted Flux": 10**predictions_log,

    "Log Residual": predictions_log - truth_log
})

prediction_df.to_csv(
    TABLE_DIR/"predictions.csv",
    index=False
)

history_df.to_csv(
    TABLE_DIR/"history.csv",
    index=False
)

pd.DataFrame([metrics]).to_csv(
    TABLE_DIR/"metrics.csv",
    index=False
)

X_test_df = pd.DataFrame(
    X_test,
    columns=[
        "Temperature",
        "Radius",
        "Distance"
    ]
)

X_test_df.to_csv(
    TABLE_DIR/"scaled_test_features.csv",
    index=False
)

with open(
    TRIAL_DIR/"architecture.txt",
    "w"
) as f:

    f.write(str(model))

pd.DataFrame(X_train).to_csv(
    TABLE_DIR/"X_train_scaled.csv",
    index=False
)

pd.DataFrame(X_test).to_csv(
    TABLE_DIR/"X_test_scaled.csv",
    index=False
)

pd.DataFrame(y_train).to_csv(
    TABLE_DIR/"y_train.csv",
    index=False
)

pd.DataFrame(y_test).to_csv(
    TABLE_DIR/"y_test.csv",
    index=False
)


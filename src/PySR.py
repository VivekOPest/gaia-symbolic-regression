"""
==============================================================
Gaia Symbolic Regression
==============================================================

Project:
    Discover an empirical symbolic equation relating stellar

        Flux = f(Temperature, Radius, Distance)

Workflow
--------

1. Load and validate dataset
2. Exploratory data analysis
3. Local exponent estimation
4. Global power-law regression
5. Residual analysis
6. Symbolic regression (PySR)
7. Final evaluation and export

Author
------
Vivek Choudhary

==============================================================
"""

# ==========================================================
# Imports
# ==========================================================

import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from pysr import PySRRegressor


# ==========================================================
# Project Directories
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "gaia_features.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ==========================================================
# Automatically Create Trial Folder
# ==========================================================

existing_trials = []

for folder in RESULTS_DIR.iterdir():

    if folder.is_dir():

        match = re.match(r"pysr_trial(\d+)", folder.name)

        if match:

            existing_trials.append(int(match.group(1)))

trial_number = max(existing_trials, default=0) + 1

TRIAL_NAME = f"pysr_trial{trial_number:03d}"

TRIAL_DIR = RESULTS_DIR / TRIAL_NAME

TRIAL_DIR.mkdir()


# ==========================================================
# Subdirectories
# ==========================================================

TABLE_DIR = TRIAL_DIR / "tables"
PLOT_DIR = TRIAL_DIR / "plots"

TABLE_DIR.mkdir()
PLOT_DIR.mkdir()


# ==========================================================
# Output Files
# ==========================================================

CONFIG_PATH = TRIAL_DIR / "config.json"

METRICS_PATH = TRIAL_DIR / "metrics.json"

SUMMARY_PATH = TRIAL_DIR / "summary.txt"

LOG_PATH = TRIAL_DIR / "analysis.log"

LINEAR_MODEL_PATH = TRIAL_DIR / "powerlaw_model.pkl"

PYSR_MODEL_PATH = TRIAL_DIR / "pysr_model.pkl"


# ==========================================================
# Global Configuration
# ==========================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

MAX_SAMPLE = 20000

LOCAL_GROUP_MIN_SIZE = 30

LOCAL_GROUP_TOLERANCE = 0.05

PYSR_ITERATIONS = 250

PYSR_POPULATIONS = 40

PYSR_POPULATION_SIZE = 40

PYSR_PARSIMONY = 0.005

DEVICE = "CPU"


# ==========================================================
# Save Configuration
# ==========================================================

config = {

    "random_state": RANDOM_STATE,

    "test_size": TEST_SIZE,

    "maximum_sample": MAX_SAMPLE,

    "local_group_min_size": LOCAL_GROUP_MIN_SIZE,

    "local_group_tolerance": LOCAL_GROUP_TOLERANCE,

    "pysr_iterations": PYSR_ITERATIONS,

    "pysr_populations": PYSR_POPULATIONS,

    "pysr_population_size": PYSR_POPULATION_SIZE,

    "pysr_parsimony": PYSR_PARSIMONY,

}

with open(CONFIG_PATH, "w") as f:

    json.dump(config, f, indent=4)


# ==========================================================
# Logging Utility
# ==========================================================

def log(message):

    print(message)

    with open(LOG_PATH, "a") as logfile:

        logfile.write(message + "\n")


# ==========================================================
# Header
# ==========================================================

log("=" * 70)
log("GAIA SYMBOLIC REGRESSION")
log("=" * 70)

log(f"Trial            : {TRIAL_NAME}")
log(f"Project Root     : {PROJECT_ROOT}")
log(f"Dataset          : {DATA_PATH}")
log(f"Results Folder   : {TRIAL_DIR}")
log(f"Device           : {DEVICE}")

log("=" * 70)

# ==========================================================
# Load and Validate Dataset
# ==========================================================

log("")
log("=" * 70)
log("PHASE 1 : DATASET LOADING & VALIDATION")
log("=" * 70)

start_phase = time.time()

# ----------------------------------------------------------
# Load CSV
# ----------------------------------------------------------

df = pd.read_csv(DATA_PATH)

log(f"Original rows : {len(df):,}")

# ----------------------------------------------------------
# Keep only required columns
# ----------------------------------------------------------

required_columns = [

    "temperature_K",
    "radius_m",
    "distance_m",
    "flux_proxy"

]

df = df[required_columns]

# ----------------------------------------------------------
# Remove NaN values
# ----------------------------------------------------------

before = len(df)

df = df.dropna()

removed = before - len(df)

log(f"Removed NaN rows : {removed}")

# ----------------------------------------------------------
# Remove infinities
# ----------------------------------------------------------

before = len(df)

df = df.replace(

    [np.inf, -np.inf],

    np.nan

)

df = df.dropna()

removed = before - len(df)

log(f"Removed infinite rows : {removed}")

# ----------------------------------------------------------
# Remove impossible values
# ----------------------------------------------------------

before = len(df)

df = df[

    (df["temperature_K"] > 0)

    &

    (df["radius_m"] > 0)

    &

    (df["distance_m"] > 0)

    &

    (df["flux_proxy"] > 0)

]

removed = before - len(df)

log(f"Removed non-positive rows : {removed}")

# ----------------------------------------------------------
# Random Sampling
# ----------------------------------------------------------

if len(df) > MAX_SAMPLE:

    df = df.sample(

        MAX_SAMPLE,

        random_state=RANDOM_STATE

    )

    log(f"Randomly sampled {MAX_SAMPLE:,} stars")

else:

    log("Dataset smaller than sampling limit")

# ----------------------------------------------------------
# Reset index
# ----------------------------------------------------------

df = df.reset_index(drop=True)

log(f"Final dataset size : {len(df):,}")

# ----------------------------------------------------------
# Compute logarithms
# ----------------------------------------------------------

df["logT"] = np.log10(df["temperature_K"])
df["logR"] = np.log10(df["radius_m"])
df["logD"] = np.log10(df["distance_m"])
df["logF"] = np.log10(df["flux_proxy"])

log("Computed logarithmic variables")

# ----------------------------------------------------------
# Save cleaned dataset
# ----------------------------------------------------------

df.to_csv(

    TABLE_DIR / "cleaned_dataset.csv",

    index=False

)

log("Saved cleaned dataset")

# ==========================================================
# Dataset Statistics
# ==========================================================

statistics = pd.DataFrame(

    {

        "Minimum": df.min(),

        "Maximum": df.max(),

        "Mean": df.mean(),

        "Median": df.median(),

        "Std": df.std(),

        "25 Percentile": df.quantile(0.25),

        "75 Percentile": df.quantile(0.75)

    }

)

statistics.to_csv(

    TABLE_DIR / "dataset_statistics.csv"

)

log("Saved dataset statistics")

# ==========================================================
# Histograms
# ==========================================================

histograms = [

    ("temperature_K", "Temperature (K)", "histogram_temperature.png"),

    ("radius_m", "Radius (m)", "histogram_radius.png"),

    ("distance_m", "Distance (m)", "histogram_distance.png"),

    ("flux_proxy", "Flux", "histogram_flux.png"),

]

for column, xlabel, filename in histograms:

    plt.figure(figsize=(8,5))

    plt.hist(

        df[column],

        bins=80

    )

    plt.xlabel(xlabel)

    plt.ylabel("Count")

    plt.title(f"Distribution of {xlabel}")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        PLOT_DIR / filename,

        dpi=300

    )

    plt.close()

log("Saved histograms")

# ==========================================================
# Dataset Summary
# ==========================================================

log("")
log("Dataset Summary")

log(f"Temperature range : {df['temperature_K'].min():.2f}  -  {df['temperature_K'].max():.2f}")

log(f"Radius range      : {df['radius_m'].min():.3e}  -  {df['radius_m'].max():.3e}")

log(f"Distance range    : {df['distance_m'].min():.3e}  -  {df['distance_m'].max():.3e}")

log(f"Flux range        : {df['flux_proxy'].min():.3e}  -  {df['flux_proxy'].max():.3e}")

phase_time = time.time() - start_phase

log(f"\nPhase completed in {phase_time:.2f} seconds")

log("=" * 70)

# ==========================================================
# Local Neighbourhood Analysis Configuration
# ==========================================================

print("=" * 60)
print("Preparing Local Neighbourhood Analysis")
print("=" * 60)

# ----------------------------------------------------------
# Number of random experiments
# ----------------------------------------------------------

NUM_EXPERIMENTS = 5000
# Increase later to 20000 or more if desired.

# ----------------------------------------------------------
# Number of neighbours used in each local regression
# ----------------------------------------------------------

NEIGHBOURS = 50

# ----------------------------------------------------------
# Random generator
# ----------------------------------------------------------

rng = np.random.default_rng(RANDOM_STATE)

# ----------------------------------------------------------
# Log-space matrix
# ----------------------------------------------------------

X_log = df[["logT","logR","logD"]].to_numpy()
# ----------------------------------------------------------
# Convenience arrays
# ----------------------------------------------------------

logT = df["logT"].to_numpy()
logR = df["logR"].to_numpy()
logD = df["logD"].to_numpy()

logF = df["logF"].to_numpy()

print(f"Experiments        : {NUM_EXPERIMENTS}")
print(f"Neighbours         : {NEIGHBOURS}")
print(f"Total Stars        : {len(logF)}")

print("=" * 60)

# ==========================================================
# Helper Function:
# Find Local Neighbourhood
# ==========================================================

def get_neighbourhood(variable, centre_index):
    """
    Returns indices of the nearest neighbouring stars while
    keeping the other two variables approximately constant.

    Parameters
    ----------
    variable : str
        "T", "R", or "D"

    centre_index : int

    Returns
    -------
    numpy.ndarray
        Indices of neighbouring stars.
    """

    if variable == "T":

        comparison = np.column_stack((logR, logD))

    elif variable == "R":

        comparison = np.column_stack((logT, logD))

    elif variable == "D":

        comparison = np.column_stack((logT, logR))

    else:

        raise ValueError("Variable must be T, R or D")

    centre = comparison[centre_index]

    distances = np.linalg.norm(
        comparison - centre,
        axis=1
    )

    order = np.argsort(distances)

    return order[:NEIGHBOURS]

# ==========================================================
# Helper Function:
# Local Power-Law Regression
# ==========================================================

def local_power_fit(variable, indices):
    """
    Fits a local power law

        logF = exponent * log(variable) + intercept

    within a local neighbourhood.

    Parameters
    ----------
    variable : str
        "T", "R", or "D"

    indices : ndarray
        Indices of neighbouring stars.

    Returns
    -------
    dict
        exponent
        intercept
        r2
        rmse
        neighbours
    """

    # ------------------------------------------------------
    # Select independent variable
    # ------------------------------------------------------

    if variable == "T":

        x = logT[indices]

    elif variable == "R":

        x = logR[indices]

    elif variable == "D":

        x = logD[indices]

    else:

        raise ValueError("Variable must be T, R or D")

    y_local = logF[indices]

    # ------------------------------------------------------
    # Fit regression
    # ------------------------------------------------------

    model = LinearRegression()

    model.fit(
        x.reshape(-1, 1),
        y_local
    )

    predictions = model.predict(
        x.reshape(-1, 1)
    )

    exponent = float(model.coef_[0])

    intercept = float(model.intercept_)

    r2 = r2_score(
        y_local,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_local,
            predictions
        )
    )

    return {

        "Exponent": exponent,

        "Intercept": intercept,

        "R2": r2,

        "RMSE": rmse,

        "Neighbours": len(indices)

    }

# ==========================================================
# Perform Local Exponent Experiments
# ==========================================================

def run_local_experiments(variable):
    """
    Performs repeated local regressions for one variable.

    Parameters
    ----------
    variable : str
        "T", "R", or "D"

    Returns
    -------
    pandas.DataFrame
        One row per experiment.
    """

    print(f"\nRunning experiments for {variable}...")

    results = []

    for experiment in range(NUM_EXPERIMENTS):

        # --------------------------------------------------
        # Randomly choose one centre star
        # --------------------------------------------------

        centre_index = rng.integers(len(logF))

        # --------------------------------------------------
        # Find neighbouring stars
        # --------------------------------------------------

        neighbours = get_neighbourhood(
            variable,
            centre_index
        )

        # --------------------------------------------------
        # Fit local power law
        # --------------------------------------------------

        fit = local_power_fit(
            variable,
            neighbours
        )

        # --------------------------------------------------
        # Store result
        # --------------------------------------------------

        fit["Experiment"] = experiment + 1
        fit["CentreIndex"] = centre_index

        results.append(fit)

        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        if (experiment + 1) % 500 == 0:

            print(
                f"{experiment+1}/{NUM_EXPERIMENTS} completed"
            )

    results = pd.DataFrame(results)

    print(
        f"Finished {variable} experiments.\n"
    )

    return results

# ==========================================================
# Run Local Exponent Experiments
# ==========================================================

print("=" * 60)
print("Running Local Exponent Discovery")
print("=" * 60)

results_tables = {}

for variable in ["T", "R", "D"]:

    print(f"\n{'='*60}")
    print(f"Estimating exponent of {variable}")
    print(f"{'='*60}")

    df_results = run_local_experiments(variable)

    # ------------------------------------------------------
    # Save immediately
    # ------------------------------------------------------

    save_path = TABLE_DIR / f"neighbourhood_results_{variable}.csv"

    df_results.to_csv(
        save_path,
        index=False
    )

    print(f"Saved -> {save_path.name}")

    results_tables[variable] = df_results

    # ------------------------------------------------------
    # Quick statistics
    # ------------------------------------------------------

    print("\nSummary")

    print(f"Mean Exponent   : {df_results['Exponent'].mean():.6f}")
    print(f"Median Exponent : {df_results['Exponent'].median():.6f}")
    print(f"Std Dev         : {df_results['Exponent'].std():.6f}")

    print(f"Mean R²         : {df_results['R2'].mean():.6f}")
    print(f"Mean RMSE       : {df_results['RMSE'].mean():.6f}")

print("\nAll local experiments completed.")

print("=" * 60)

# ==========================================================
# Statistical Analysis of Local Exponents
# ==========================================================

print("=" * 60)
print("Analysing Local Exponents")
print("=" * 60)

statistics_rows = []

weighted_exponents = {}

BOOTSTRAP_ITERATIONS = 1000

rng = np.random.default_rng(RANDOM_STATE)

for variable, df_results in results_tables.items():

    exponents = df_results["Exponent"].to_numpy()

    r2 = df_results["R2"].to_numpy()

    rmse = df_results["RMSE"].to_numpy()

    # ------------------------------------------------------
    # Construct weights
    # ------------------------------------------------------

    weights = r2 / (rmse + 1e-8)

    weights = np.maximum(weights, 0)

    weights /= np.sum(weights)

    weighted_mean = np.sum(weights * exponents)

    weighted_variance = np.sum(
        weights * (exponents - weighted_mean) ** 2
    )

    weighted_std = np.sqrt(weighted_variance)

    weighted_exponents[variable] = weighted_mean

    # ------------------------------------------------------
    # Bootstrap confidence interval
    # ------------------------------------------------------

    bootstrap_means = []

    for _ in range(BOOTSTRAP_ITERATIONS):

        indices = rng.choice(
            len(exponents),
            size=len(exponents),
            replace=True,
            p=weights
        )

        bootstrap_means.append(
            np.mean(exponents[indices])
        )

    lower = np.percentile(
        bootstrap_means,
        2.5
    )

    upper = np.percentile(
        bootstrap_means,
        97.5
    )

    # ------------------------------------------------------
    # Save statistics
    # ------------------------------------------------------

    statistics_rows.append({

        "Variable": variable,

        "Mean Exponent":
            np.mean(exponents),

        "Median Exponent":
            np.median(exponents),

        "Weighted Mean":
            weighted_mean,

        "Weighted Std":
            weighted_std,

        "Minimum":
            np.min(exponents),

        "Maximum":
            np.max(exponents),

        "95% CI Lower":
            lower,

        "95% CI Upper":
            upper,

        "Mean R2":
            np.mean(r2),

        "Mean RMSE":
            np.mean(rmse),

        "Experiments":
            len(df_results)

    })

    print(f"\n{variable}")

    print(f"Weighted Mean : {weighted_mean:.6f}")

    print(f"95% CI        : [{lower:.6f}, {upper:.6f}]")

    print(f"Weighted Std  : {weighted_std:.6f}")

statistics_df = pd.DataFrame(statistics_rows)

statistics_df.to_csv(

    TABLE_DIR / "exponent_statistics.csv",

    index=False

)

pd.DataFrame({

    "Variable": list(weighted_exponents.keys()),

    "Exponent": list(weighted_exponents.values())

}).to_csv(

    TABLE_DIR / "weighted_exponents.csv",

    index=False

)

print("\nExponent statistics saved.")

# ==========================================================
# Global Power-law Regression
# ==========================================================

log("\n")
log("=" * 70)
log("GLOBAL POWER-LAW REGRESSION")
log("=" * 70)

# ----------------------------------------------------------
# Starting point from neighbourhood analysis
# ----------------------------------------------------------

initial_guess = np.array([
    weighted_exponents["T"],
    weighted_exponents["R"],
    weighted_exponents["D"],
    0.0
])

log("Initial exponents from neighbourhood analysis:")

log(f"T : {initial_guess[0]:.6f}")
log(f"R : {initial_guess[1]:.6f}")
log(f"D : {initial_guess[2]:.6f}")

# ----------------------------------------------------------
# Objective Function
# ----------------------------------------------------------

def residual_function(params):

    a, b, c, d = params

    prediction = (

        a * df["logT"].to_numpy()

        + b * df["logR"].to_numpy()

        + c * df["logD"].to_numpy()

        + d

    )

    return prediction - df["logF"].to_numpy()

# ----------------------------------------------------------
# Global optimisation
# ----------------------------------------------------------

result = least_squares(

    residual_function,

    x0=initial_guess,

    method="trf"

)

a, b, c, d = result.x

log("\nOptimised exponents")

log(f"T exponent : {a:.6f}")

log(f"R exponent : {b:.6f}")

log(f"D exponent : {c:.6f}")

log(f"Intercept  : {d:.6f}")

log(f"Iterations : {result.nfev}")

log(f"Success    : {result.success}")

# ----------------------------------------------------------
# Construct transformed variables
# ----------------------------------------------------------

df["TemperatureTerm"] = a * df["logT"]
df["RadiusTerm"] = b * df["logR"]
df["DistanceTerm"] = c * df["logD"]

pd.DataFrame({

    "Variable":["T","R","D"],

    "Initial":[
        weighted_exponents["T"],
        weighted_exponents["R"],
        weighted_exponents["D"]
    ],

    "Optimised":[a,b,c]

}).to_csv(

    TABLE_DIR / "global_exponents.csv",

    index=False

)


# ==========================================================
# Guided Symbolic Regression (PySR)
# ==========================================================

log("\n")
log("=" * 70)
log("GUIDED SYMBOLIC REGRESSION")
log("=" * 70)

log("Preparing variables for PySR...")

# ----------------------------------------------------------
# Features
# ----------------------------------------------------------

X_pysr = pd.DataFrame({

    "TemperatureTerm": df["TemperatureTerm"],
    "RadiusTerm": df["RadiusTerm"],
    "DistanceTerm": df["DistanceTerm"]

})

y_pysr = np.log10(df["flux_proxy"])

# ----------------------------------------------------------
# Train/Test split
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X_pysr,
    y_pysr,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    shuffle=True

)

log(f"Training samples : {len(X_train)}")
log(f"Testing samples  : {len(X_test)}")

# ----------------------------------------------------------
# Build model
# ----------------------------------------------------------

log("\nBuilding PySR model...")

model = PySRRegressor(

    niterations=PYSR_ITERATIONS,
    populations=PYSR_POPULATIONS,
    population_size=PYSR_POPULATION_SIZE,

    random_state=RANDOM_STATE,

    model_selection="best",

    parsimony=0.02,

    # Equation size
    maxsize=10,
    maxdepth=10,

    # Operators
    binary_operators=[
        "+",
        "*"
    ],

    unary_operators=[],

    # Loss
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",

    verbosity=1,

    batching=True,
    batch_size=2048,

    procs=2
)

log("Training PySR...")

start = time.time()

model.fit(

    X_train,
    y_train,

    variable_names=[

        "TemperatureTerm",
        "RadiusTerm",
        "DistanceTerm"

    ]

)

training_time = time.time() - start

log(f"Training completed in {training_time:.1f} s")

# ----------------------------------------------------------
# Save model
# ----------------------------------------------------------

joblib.dump(

    model,

    PYSR_MODEL_PATH

)

equations = model.equations_

equations.to_csv(

    TABLE_DIR / "equations.csv",

    index=False

)

log(f"{len(equations)} candidate equations saved.")


# ==========================================================
# Evaluate Symbolic Regression Model
# ==========================================================

log("\n")
log("=" * 70)
log("MODEL EVALUATION")
log("=" * 70)

# ----------------------------------------------------------
# Predictions
# ----------------------------------------------------------

predictions = model.predict(X_test)

truth = y_test.to_numpy()

residuals = predictions - truth

# ----------------------------------------------------------
# Metrics
# ----------------------------------------------------------

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

metrics = {

    "Training Samples": len(X_train),

    "Testing Samples": len(X_test),

    "Training Time (s)": float(training_time),

    "RMSE": float(rmse),

    "MAE": float(mae),

    "R2": float(r2),

    "Residual Mean": float(np.mean(residuals)),

    "Residual Std": float(np.std(residuals)),

    "Residual Median": float(np.median(residuals)),

    "Residual Max": float(np.max(residuals)),

    "Residual Min": float(np.min(residuals))

}

with open(

    METRICS_PATH,

    "w"

) as f:

    json.dump(

        metrics,

        f,

        indent=4

    )

log(f"RMSE : {rmse:.6f}")
log(f"MAE  : {mae:.6f}")
log(f"R²   : {r2:.6f}")

# ----------------------------------------------------------
# Save Predictions
# ----------------------------------------------------------

prediction_df = pd.DataFrame({

    "True log10 Flux": truth,

    "Predicted log10 Flux": predictions,

    "Residual": residuals,

    "True Flux": 10**truth,

    "Predicted Flux": 10**predictions

})

prediction_df.to_csv(

    TABLE_DIR / "predictions.csv",

    index=False

)

log("Prediction table saved.")

# ----------------------------------------------------------
# Best Equation
# ----------------------------------------------------------

best = model.get_best()

log("\n")
log("=" * 70)
log("BEST EQUATION")
log("=" * 70)

print(best)

log(str(best))

# ----------------------------------------------------------
# Save Equation Table
# ----------------------------------------------------------

equations = model.equations_

equations.to_csv(

    TABLE_DIR / "equations.csv",

    index=False

)

# ----------------------------------------------------------
# Save Human Readable Equations
# ----------------------------------------------------------

with open(

    TRIAL_DIR / "equations.txt",

    "w"

) as f:

    f.write("Candidate Equations\n")

    f.write("="*70 + "\n\n")

    for i, row in equations.iterrows():

        f.write(f"Equation {i+1}\n")
        f.write("-"*40 + "\n")

        f.write(f"Complexity : {row['complexity']}\n")
        f.write(f"Loss       : {row['loss']}\n")

        if "score" in row.index:

            f.write(f"Score      : {row['score']}\n")

        f.write(f"Equation   : {row['equation']}\n\n")

log("Equation table saved.")

# ----------------------------------------------------------
# Save Summary
# ----------------------------------------------------------

with open(

    SUMMARY_PATH,

    "w"

) as f:

    f.write("GAIA SYMBOLIC REGRESSION\n")

    f.write("="*60 + "\n\n")

    f.write(f"Training Samples : {len(X_train)}\n")
    f.write(f"Testing Samples  : {len(X_test)}\n")
    f.write(f"Training Time    : {training_time:.2f} s\n\n")

    f.write(f"RMSE : {rmse:.6f}\n")
    f.write(f"MAE  : {mae:.6f}\n")
    f.write(f"R²   : {r2:.6f}\n\n")

    f.write("Best Equation\n")
    f.write("-"*40 + "\n")
    f.write(str(best))

log("Summary written.")


# ==========================================================
# Figures
# ==========================================================

log("\n")
log("=" * 70)
log("GENERATING FIGURES")
log("=" * 70)

# ----------------------------------------------------------
# Prediction vs Truth
# ----------------------------------------------------------

plt.figure(figsize=(8,8))

plt.scatter(

    truth,
    predictions,

    s=8,
    alpha=0.45

)

minimum = min(truth.min(), predictions.min())
maximum = max(truth.max(), predictions.max())

plt.plot(

    [minimum, maximum],
    [minimum, maximum],

    "r--",

    linewidth=2

)

plt.xlabel("True log10 Flux")
plt.ylabel("Predicted log10 Flux")

plt.title("Prediction vs Truth")

plt.grid(True)

plt.tight_layout()

plt.savefig(

    PLOT_DIR / "prediction_vs_truth.png",

    dpi=300

)

plt.close()

log("Prediction plot saved.")

# ----------------------------------------------------------
# Residual Histogram
# ----------------------------------------------------------

plt.figure(figsize=(9,6))

plt.hist(

    residuals,

    bins=70

)

plt.xlabel("Residual")

plt.ylabel("Count")

plt.title("Residual Distribution")

plt.grid(True)

plt.tight_layout()

plt.savefig(

    PLOT_DIR / "residual_distribution.png",

    dpi=300

)

plt.close()

log("Residual histogram saved.")

# ----------------------------------------------------------
# Residual vs Truth
# ----------------------------------------------------------

plt.figure(figsize=(9,6))

plt.scatter(

    truth,

    residuals,

    s=8,

    alpha=0.45

)

plt.axhline(

    0,

    color="red",

    linestyle="--"

)

plt.xlabel("True log10 Flux")

plt.ylabel("Residual")

plt.title("Residual vs Truth")

plt.grid(True)

plt.tight_layout()

plt.savefig(

    PLOT_DIR / "residual_vs_truth.png",

    dpi=300

)

plt.close()

log("Residual plot saved.")

# ----------------------------------------------------------
# Pareto Frontier
# ----------------------------------------------------------

plt.figure(figsize=(9,6))

plt.plot(

    equations["complexity"],

    equations["loss"],

    marker="o",

    linewidth=2,

    markersize=5

)

plt.xlabel("Equation Complexity")

plt.ylabel("Loss")

plt.title("Pareto Frontier")

plt.grid(True)

plt.tight_layout()

plt.savefig(

    PLOT_DIR / "pareto_frontier.png",

    dpi=300

)

plt.close()

log("Pareto Frontier saved.")

# ----------------------------------------------------------
# Complexity vs Score
# ----------------------------------------------------------

if "score" in equations.columns:

    plt.figure(figsize=(9,6))

    plt.plot(

        equations["complexity"],

        equations["score"],

        marker="o",

        linewidth=2,

        markersize=5

    )

    plt.xlabel("Equation Complexity")

    plt.ylabel("Equation Score")

    plt.title("Equation Score")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        PLOT_DIR / "equation_score.png",

        dpi=300

    )

    plt.close()

    log("Equation score plot saved.")

# ----------------------------------------------------------
# Distribution of Local Exponents
# ----------------------------------------------------------

plt.figure(figsize=(9,6))

plt.hist(

    results_tables["T"]["Exponent"],

    bins=35,

    alpha=0.5,

    label="Temperature"

)

plt.hist(

    results_tables["R"]["Exponent"],

    bins=35,

    alpha=0.5,

    label="Radius"

)

plt.hist(

    results_tables["D"]["Exponent"],

    bins=35,

    alpha=0.5,

    label="Distance"

)

plt.xlabel("Estimated Exponent")

plt.ylabel("Frequency")

plt.title("Distribution of Local Power-law Exponents")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(

    PLOT_DIR / "local_exponent_distribution.png",

    dpi=300

)

plt.close()

log("Exponent distribution saved.")

log("All figures generated.")

# ==========================================================
# Statistical Analysis of Local Exponents
# ==========================================================

log("\n")
log("=" * 70)
log("LOCAL EXPONENT STATISTICS")
log("=" * 70)

statistics = []

for variable in ["T", "R", "D"]:

    values = results_tables[variable]["Exponent"].dropna()

    mean = values.mean()

    median = values.median()

    std = values.std()

    minimum = values.min()

    maximum = values.max()

    q1 = values.quantile(0.25)

    q3 = values.quantile(0.75)

    ci95 = 1.96 * std / np.sqrt(len(values))

    cv = std / abs(mean)

    statistics.append({

        "Variable": variable,

        "Samples": len(values),

        "Mean": mean,

        "Median": median,

        "Std": std,

        "Min": minimum,

        "Max": maximum,

        "Q1": q1,

        "Q3": q3,

        "95% CI": ci95,

        "Coefficient of Variation": cv

    })

    log("")
    log(f"{variable}")
    log("-"*40)
    log(f"Mean     : {mean:.6f}")
    log(f"Median   : {median:.6f}")
    log(f"Std Dev  : {std:.6f}")
    log(f"95% CI   : ±{ci95:.6f}")
    log(f"Min      : {minimum:.6f}")
    log(f"Max      : {maximum:.6f}")
    log(f"CV       : {cv:.6f}")

statistics_df = pd.DataFrame(statistics)

statistics_df.to_csv(

    TABLE_DIR / "local_exponent_statistics.csv",

    index=False

)

statistics_df.to_json(

    TABLE_DIR / "local_exponent_statistics.json",

    orient="records",

    indent=4

)

log("")
log("Exponent statistics saved.")

# ==========================================================
# Mean Local Exponents with 95% Confidence Intervals
# ==========================================================

plt.figure(figsize=(8,6))

variables = statistics_df["Variable"]

means = statistics_df["Mean"]

errors = statistics_df["95% CI"]

plt.errorbar(

    variables,

    means,

    yerr=errors,

    fmt="o",

    capsize=6,

    linewidth=2

)

plt.grid(True)

plt.xlabel("Variable")

plt.ylabel("Estimated Exponent")

plt.title("Mean Local Exponents (95% Confidence Interval)")

plt.tight_layout()

plt.savefig(

    PLOT_DIR / "local_exponent_confidence.png",

    dpi=300

)

plt.close()

log("Confidence interval plot saved.")
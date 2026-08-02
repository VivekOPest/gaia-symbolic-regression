"""
model.py

Neural Network for predicting Gaia Flux Proxy
from stellar properties.

Inputs:
    Temperature (K)
    Radius (m)
    Distance (m)
    Physics Predictor (R²T⁴/d²)

Output:
    Gaia Flux Proxy
"""

import os
import time
import joblib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset
from torch.utils.data import DataLoader




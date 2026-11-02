"""Data loading and train-test partitioning utilities."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "MedHouseVal"
FEATURE_COLS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

def load_housing_data(csv_path="data/california_housing.csv"):
    """Load the raw California Housing dataset from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run download_data.py first.")
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return df, X, y

def get_train_test_data(csv_path="data/california_housing.csv", test_size=0.2, random_state=42):
    """Split into training (80%) and untouched test (20%) partitions."""
    df, X, y = load_housing_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

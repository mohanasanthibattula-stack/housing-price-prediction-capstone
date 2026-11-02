"""Download and save the California Housing dataset."""
import os
import pandas as pd
from sklearn.datasets import fetch_california_housing

def main():
    os.makedirs("data", exist_ok=True)
    csv_path = "data/california_housing.csv"
    
    print("Fetching California Housing dataset from scikit-learn...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print("Sample rows:")
    print(df.head(3))
    
    df.to_csv(csv_path, index=False)
    print(f"Saved raw data to {csv_path} ({os.path.getsize(csv_path)} bytes)")

if __name__ == "__main__":
    main()

# preprocess_datasets.py

import pandas as pd
import os

def preprocess_csv(file_path):
    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.to_csv(file_path, index=False)

def preprocess_datasets(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            preprocess_csv(os.path.join(directory, filename))

if __name__ == "__main__":
    preprocess_datasets('datasets')

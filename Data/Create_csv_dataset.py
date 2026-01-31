import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "wdbc.data")


# define column names
columns = [
    "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean",
    "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"
] + [f"feature_{i}" for i in range(11, 31)]  # there are a total of 30 numeric features after diagnosis

# load CSV
df = pd.read_csv(DATA_PATH, header=None, names=columns)

df.to_csv(os.path.join(BASE_DIR, "dataset.csv"), index=False)

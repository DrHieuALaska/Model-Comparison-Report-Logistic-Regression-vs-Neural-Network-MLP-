import matplotlib.pyplot as plt
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")
df = pd.read_csv(DATA_PATH)

# take full rows, build histograms for features

# df.iloc[:, [12,13,14,15]].hist(bins=30, figsize=(10,6))
# plt.tight_layout()
# plt.show()


# plot correlation heatmap

corr = df.drop(columns=['id', 'diagnosis']).corr()
import seaborn as sns
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

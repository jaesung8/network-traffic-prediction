import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.graph_objects as go

def detect_and_remove_outliers_zscore(df, threshold=3):
    outlier_indices = {}

    for column in df.columns:
        if column != 'time':  # 'time' 열 제외
            mean = df[column].mean()
            std = df[column].std()
            z_scores = (df[column] - mean) / std

            outliers = np.where(np.abs(z_scores) > threshold)[0]
            outlier_indices[column] = outliers

    return outlier_indices

file_path = 'data/geant_traffic.csv'
# file_path = 'data/abilene/abilene_traffic.csv'
df = pd.read_csv(file_path, index_col='Datetime')
print(df)

z_outliers = detect_and_remove_outliers_zscore(df)

def plot_heatmap(_data):
    plt.figure(figsize=(12, 6))
    sns.heatmap(_data.T, cmap="viridis", cbar=True, xticklabels=10, yticklabels=nodes)
    plt.title("Time-Series Data for Nodes (Heatmap)")
    plt.xlabel("Time Steps")
    plt.ylabel("Nodes")
    plt.savefig('heatmap.png', dpi=300, bbox_inches="tight")


def load_and_plot(data):
    plt.figure(figsize=(12, 6))

    for column in data.columns:
        print(column)
        plt.plot(data.index, data[column], label=column)
        break

    # plt.title('Traffics Over Time')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot.png', dpi=300, bbox_inches="tight")

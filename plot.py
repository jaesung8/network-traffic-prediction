import os
from collections import defaultdict

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
            print(column, outliers)
            outlier_indices[column] = outliers

    return outlier_indices


def plot_heatmap(_data):
    plt.figure(figsize=(12, 6))
    sns.heatmap(_data.T, cmap="viridis", cbar=True, xticklabels=10, yticklabels=nodes)
    plt.title("Time-Series Data for Nodes (Heatmap)")
    plt.xlabel("Time Steps")
    plt.ylabel("Nodes")
    plt.savefig('heatmap.png', dpi=300, bbox_inches="tight")


def load_and_plot():
    npy_files = [f for f in os.listdir('./pred_values') if f.endswith('.npy')]
    data_dict = defaultdict(list)
    seq_len = 100
    seq_len = 80
    timeoffset = 40

    # 각 npy 파일의 데이터를 읽어서 저장
    for file in npy_files:
    # 파일명에서 dataset_name과 model_name 추출
        try:
            _, _, dataset_name, model_name = file.replace('.npy', '').split('_')
        except ValueError:
            print(f"Skipping file {file} due to incorrect format.")
            continue
        
        data = np.load(os.path.join('./pred_values', file))
        data_dict[dataset_name].append((model_name, data))

    for dataset_name, models in data_dict.items():
        plt.figure(figsize=(10, 6))
        if dataset_name == 'aiblene':
            step_index = 2
        else:
            step_index = 0

        for model_name, data in models:
            if model_name in ['transformer', 'dcrnn', 'arima']:
                continue
            # geant
            
            if model_name == 'gt':
                time_series = data[step_index, -seq_len-10:-seq_len + timeoffset - 10, 0]
            elif model_name == 'stdmae':
                time_series = data[step_index, -seq_len:-seq_len + timeoffset, 0]
            elif model_name == 'arima':
                time_series = data[0, -seq_len:-seq_len + timeoffset, 0]
            else:
                time_series = data[step_index, -seq_len-9:-seq_len + timeoffset-9, 0]

            plt.plot(time_series, label=model_name)  # 모델 데이터를 플롯
            print(model_name)
            print(data.shape)
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.title(f'Time Series Data for Dataset: {dataset_name.title()}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'timeseries/_{dataset_name}_{seq_len}_{timeoffset}.png')


def plot_test_dataset(data):
    y_data = data['y']
    print(y_data.shape)
    # (2150, 12, 72, 2)
    # (9615, 12, 30, 2)
    plt.figure(figsize=(12, 6))
    plot_num = 1000
    # plt.plot(range(plot_num), y_data[:, 2, 12, 0][-plot_num:], label=5)
    # for node_num in range(y_data.shape[2])[:10]:
    #     plt.plot(range(plot_num), y_data[:, 2, node_num, 0][-plot_num:], label=node_num)
    plt.plot(range(plot_num), y_data[:, 2, 5, 0][-plot_num:], label=5)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('geant_plot.png', dpi=300, bbox_inches="tight")


# 이상치 탐색
# file_path = 'data/abilene_traffic.csv'
# df = pd.read_csv(file_path, index_col='Datetime')
# print(df)

# z_outliers = detect_and_remove_outliers_zscore(df)


# test_data = np.load('geant/test.npz')
# print(test_data.shape)
# plot_numpy(test_data)

load_and_plot()
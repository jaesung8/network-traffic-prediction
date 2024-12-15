from collections import defaultdict
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from constants import geant, abilene, capacity

# Initialize scaler
scaler = MinMaxScaler()


def load_numpy():
    npy_files = [f for f in os.listdir('./pred_values') if f.endswith('.npy')]
    data_dict = defaultdict(list)
    seq_len = 100

    # 각 npy 파일의 데이터를 읽어서 저장
    for file in npy_files:
    # 파일명에서 dataset_name과 model_name 추출
        try:
            _, _, dataset_name, model_name = file.replace('.npy', '').split('_')
        except ValueError:
            print(f"Skipping file {file} due to incorrect format.")
            continue
        
        data = np.load(os.path.join('./pred_values', file))
        if len(data.shape) == 4:
            data = np.squeeze(data, axis=-1)
        data_dict[dataset_name].append((model_name, data))
    return data_dict

def calculate_congestion_and_cost(models_predictions, ground_truth, module_capacity, module_cost):
    max_predicted_traffic = np.max(np.array(models_predictions), axis=0)
    required_modules = np.ceil(max_predicted_traffic / module_capacity).astype(int)
    installed_capacity = required_modules * module_capacity

    congestion_events = np.where(ground_truth > installed_capacity, 1, 0)
    total_cost = np.sum(required_modules) * module_cost
    results = {
        "congestion_events_per_link": congestion_events,
        "total_congestion_events": np.sum(congestion_events),
        "total_cost": total_cost,
        "installed_modules_per_link": required_modules,
    }
    return results


def calculate_network_congestion_and_cost(models_predictions, ground_truth, module_capacity, module_cost):
    max_predicted_traffic = np.max(np.array(models_predictions), axis=0)
    required_modules = np.ceil(max_predicted_traffic / module_capacity).astype(int)
    # print('required_modules', required_modules.shape)
    installed_capacity = required_modules * module_capacity
    # print('installed_capacity', installed_capacity.shape)
    congestion_events_per_link = np.where(ground_truth > installed_capacity, 1, 0)
    # print('congestion_events_per_link',congestion_events_per_link.shape)
    total_congestion_events = np.sum(congestion_events_per_link)
    total_modules_installed = np.sum(required_modules)
    total_cost = required_modules * module_cost
    total_cost = np.sum(total_cost)

    # Step 7: Return network-wide results
    results = {
        "total_congestion_events": total_congestion_events,
        "total_cost": total_cost,
        "total_modules_installed": total_modules_installed,
    }
    return results


def plot(data, scale):
    for key, value in data.items():
        df = pd.DataFrame(value).T
        # df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        plt.figure(figsize=(8, 6))
        df[['total_congestion_events', 'total_cost']].plot(
            kind='bar',
            title=f'{key.title()} Dataset Metrics',
            ylabel='Value',
            xlabel='Model',
            width=0.4
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'cost_graph/{key}_{scale}_cost.png')

    # # Geant dataset
    # plt.figure(figsize=(12, 6))
    # geant_df.plot(
    #     kind='bar',
    #     title='Geant Dataset Metrics',
    #     ylabel='Values',
    #     xlabel='Model',
    #     figsize=(12, 6)
    # )
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()


def plot_multi_yscale(data):
    # methods = list(data.keys())
    # total_congestion_events = [data[method]['total_congestion_events'] for method in methods]
    # total_cost = [data[method]['total_cost'] for method in methods]
    # total_modules_installed = [data[method]['total_modules_installed'] for method in methods]

    # # Create a figure and axis
    # fig, ax1 = plt.subplots()

    # # Plot total congestion events on the first y-axis
    # ax1.set_xlabel('Methods')
    # ax1.set_ylabel('Total Congestion Events', color='tab:red')
    # # ax1.bar(methods, total_congestion_events, color='tab:red', alpha=0.6, label='Total Congestion Events', width=0.4)
    # ax1.plot(methods, total_congestion_events, color='tab:red', marker='o', label='Total Congestion Events')
    # ax1.tick_params(axis='y', labelcolor='tab:red')

    # # Create a second y-axis for total cost
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Total Cost', color='tab:blue')
    # ax2.plot(methods, total_cost, color='tab:blue', marker='o', label='Total Cost')
    # # ax2.bar(methods, total_cost, color='tab:blue', alpha=0.6, label='Total Cost', width=0.4)
    # ax2.tick_params(axis='y', labelcolor='tab:blue')

    # # # Create a third y-axis for total modules installed
    # ax3 = ax1.twinx()
    # # # Offset the third y-axis to avoid overlap with the second y-axis
    # ax3.spines['right'].set_position(('outward', 60))
    # ax3.set_ylabel('Total Modules Installed', color='tab:green')
    # ax3.bar(methods, total_modules_installed, color='tab:green',  alpha=0.6, label='Total Modules Installed', width=0.4)
    # ax3.tick_params(axis='y', labelcolor='tab:green')

    # # Show the plot
    # plt.title('Comparison of cost and conngestion')
    # fig.tight_layout()
    # plt.savefig(f'cost_graph/multi_cost.png')


    methods = list(data.keys())
    total_congestion_events = [data[method]['total_congestion_events'] for method in methods]
    total_cost = [data[method]['total_cost'] for method in methods]
    total_modules_installed = [data[method]['total_modules_installed'] for method in methods]

    # Create a figure and axis
    fig, ax1 = plt.subplots()

    # Plot total congestion events on the first y-axis
    ax1.set_xlabel('Methods')
    ax1.set_ylabel('Total Congestion Events', color='tab:red')
    ax1.bar(methods, total_congestion_events, color='tab:red', alpha=0.6, label='Total Congestion Events')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Create a second y-axis for total cost
    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Cost', color='tab:blue')
    ax2.plot(methods, total_cost, color='tab:blue', marker='o', label='Total Cost')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Create a third y-axis for total modules installed
    # ax3 = ax1.twinx()
    # # Offset the third y-axis to avoid overlap with the second y-axis
    # ax3.spines['right'].set_position(('outward', 60))
    # ax3.set_ylabel('Total Modules Installed', color='tab:green')
    # ax3.plot(methods, total_modules_installed, color='tab:green', marker='s', label='Total Modules Installed')
    # ax3.tick_params(axis='y', labelcolor='tab:green')

    # Show the plot
    # plt.title('Comparison of Methods from Dictionary Data')
    fig.tight_layout()
    plt.savefig(f'cost_graph/_multi_cost.png')




def main():
    import pprint

    data_dict = load_numpy()

    geant_cost = np.array(geant)
    abilene_cost = np.array(abilene)

    scale = 40000.0
    # scale = 1.0

    module_capacity = 40000.0 / scale
    module_cost = {
        'abilene': abilene_cost / scale,
        'geant': geant_cost / scale,
    }

    results_dict = {}
    for dataset_name, models in data_dict.items():
        if dataset_name == 'abilene':
            step_index = 2
        elif dataset_name == 'geant':
            step_index = 0

        results_per_dataset = {}
        for model_name, model_predictions in models:
            if model_name == 'gt':
                ground_truth = model_predictions[step_index]

        for model_name, model_predictions in models:
            if model_name in ['transformer', 'gt', 'arima', 'dcrnn']:
                continue
            results_per_model = calculate_network_congestion_and_cost(
                model_predictions[step_index], ground_truth, module_capacity, module_cost[dataset_name]
            )
            results_per_dataset[model_name] = results_per_model
            # break
        results_dict[dataset_name] = results_per_dataset

    pprint.pprint(results_dict)
    plot(results_dict, scale)
    plot_multi_yscale(results_dict['geant'])



if __name__ == "__main__":
    main()
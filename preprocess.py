import glob
import argparse
from datetime import datetime
import os
from collections import defaultdict
import re
import math
from itertools import combinations
import pickle

import networkx as nx
import pandas as pd
import numpy as np


def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth."""
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def preprocess(network_name):
    file_path = f"data/{network_name}.txt"
    with open(file_path, "r") as file:
        data = file.read()

    nodes_pattern = re.compile(r"NODES\s*\((.*?)\n\)", re.DOTALL)
    links_pattern = re.compile(r"LINKS\s*\((.*?)\n\)", re.DOTALL)

    # 섹션별 내용 추출
    nodes_match = nodes_pattern.search(data)
    links_match = links_pattern.search(data)

    # 결과 저장
    nodes_content = nodes_match.group(1).strip() if nodes_match else None
    links_content = links_match.group(1).strip() if links_match else None
    
    nodes = re.findall(r"([a-zA-Z0-9.]+)\s+\(\s+([-?\d.]+)\s+([-?\d.]+)\s+\)", nodes_content)
    node_coords = {node: (float(lat), float(lon)) for node, lat, lon in nodes}

    links = re.findall(r"\(\s+([a-zA-Z0-9.]+)\s+([a-zA-Z0-9.]+)\s+\)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\(\s+[\d.]+\s+[\d.]+\s+\)", links_content)

    print(len(nodes), len(links))
    # Initialize graph and populate
    graph = nx.Graph()
    transformed_graph = nx.Graph()
    graph.add_nodes_from(nodes)

    for src, dest in links:
        distance = haversine(*node_coords[src], *node_coords[dest])
        graph.add_edge(src, dest, weight=distance)
        transformed_graph.add_node(f'{src}_{dest}')
        transformed_graph.add_node(f'{dest}_{src}')

    for edge1, edge2 in combinations(graph.edges(data=False), 2):
        shared_node = set(edge1) & set(edge2)
        if shared_node:
            mean_distance = (graph[edge1[0]][edge1[1]]['weight'] + graph[edge2[0]][edge2[1]]['weight']) / 2
            transformed_graph.add_edge(f"{edge1[0]}_{edge1[1]}", f"{edge2[0]}_{edge2[1]}", weight=mean_distance)
            transformed_graph.add_edge(f"{edge1[1]}_{edge1[0]}", f"{edge2[1]}_{edge2[0]}", weight=mean_distance)

    file_list = glob.glob(f'data/{network_name}_raw/*.txt')
    file_list.sort()
    rows, times = [], []

    count = 0
    for i, file_path in enumerate(file_list):
        _time = parse_filename_to_datetime(file_path)

        with open(file_path, "r") as file:
            data = file.read()

        demands_pattern = re.compile(r"DEMANDS\s*\((.*?)\n\)", re.DOTALL)
        demands_match = demands_pattern.search(data)
        demands_content = demands_match.group(1).strip() if demands_match else ''

        demands = re.findall(r"\(\s+([a-zA-Z0-9\.]+)\s+([a-zA-Z0-9\.]+)\s+\)\s+\d+\s+([\d.]+)\s+", demands_content)
        if not demands:
            continue

        count += 1
        if network_name == 'geant' and count == 2188:
            print(file_path)
            continue

        # Calculate traffic per node based on demands
        node_traffic = defaultdict(float)
        for src, dest, demand in demands:
            demand = float(demand)
            path = nx.shortest_path(graph, source=src, target=dest)
            # for node in path:
            #     node_traffic[node] += demand
            num_hop = len(path)
            for i in range(num_hop - 1):
                node_traffic[f'{path[i]}_{path[i+1]}'] += demand

        # Add the traffic data as a row, filling missing nodes with 0 traffic
        # row = {node: node_traffic[node] for node in nodes}
        row = {link_name: node_traffic[link_name] for link_name in transformed_graph.nodes}
        rows.append(row)
        times.append(_time)

    # Create a DataFrame where each row corresponds to a file
    traffic_df = pd.DataFrame(rows, index=times)
    traffic_df.index.name = "Datetime"
    print(traffic_df)
    traffic_df.to_csv(f"data/{network_name}_traffic.csv")

    with open(f'data/{network_name}_adj.pkl', "wb") as f:
        pickle.dump(nx.to_numpy_array(transformed_graph, weight=None), f)
    with open(f'data/{network_name}_distance.pkl', "wb") as f:
        pickle.dump(nx.to_numpy_array(transformed_graph, weight='weight'), f)
    with open(f'data/{network_name}_norm_distance.pkl', "wb") as f:
        dist_mx = nx.to_numpy_array(transformed_graph, weight='weight')
        dist_mx = np.where(dist_mx == 0, np.inf, dist_mx)
        distances = dist_mx[~np.isinf(dist_mx)].flatten()
        std = distances.std()
        adj_mx = np.exp(-np.square(dist_mx / std))
        adj_mx[adj_mx < 0.1] = 0
        sensor_ids = transformed_graph.nodes
        num_sensors = len(sensor_ids)
        sensor_id_to_ind = {}
        for i, sensor_id in enumerate(sensor_ids):
            sensor_id_to_ind[sensor_id] = i
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f)

def parse_filename_to_datetime(filename):
    # Regular expression to capture the date and time components
    match = re.search(r"demandMatrix-\S+-(\d{8})-(\d{4})\.txt", filename)
    
    if match:
        date_part = match.group(1)  # YYYYMMDD
        time_part = match.group(2)  # HHMM
        
        # Combine and parse into a datetime object
        parsed_datetime = datetime.strptime(f"{date_part} {time_part}", "%Y%m%d %H%M")
        return parsed_datetime
    else:
        raise ValueError("Filename format does not match the expected pattern.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network-name', required=True)
    args = parser.parse_args()
    preprocess(args.network_name)

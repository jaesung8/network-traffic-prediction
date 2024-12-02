import glob
import argparse
from datetime import datetime
import os
from collections import defaultdict
import re

import networkx as nx
import pandas as pd


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
    
    nodes = re.findall(r"([a-zA-Z0-9.]+)\s+\(\s+[-\d.]+\s+[-\d.]+\s+\)", nodes_content)
    links = re.findall(r"\(\s+([a-zA-Z0-9.]+)\s+([a-zA-Z0-9.]+)\s+\)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\(\s+[\d.]+\s+[\d.]+\s+\)", links_content)
    # r"([A-Z]+\w+_[A-Z]+\w+\s+\(\s+[A-Z]+\w+\s+[A-Z]+\w+\s+\)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\(\s+[\d.]+\s+[\d.]+\s+\)$"
    print(len(nodes), len(links))
    # Initialize graph and populate
    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    link_names = []
    for src, dest in links:
        graph.add_edge(src, dest)
        link_names.append(f'{src}_{dest}')
        link_names.append(f'{dest}_{src}')

    file_list = glob.glob(f'data/{network_name}/*.txt')
    file_list.sort()
    rows, times = [], []
    print(len(file_list))
    for i, file_path in enumerate(file_list):
        _time = parse_filename_to_datetime(file_path)

        with open(file_path, "r") as file:
            data = file.read()

        demands_pattern = re.compile(r"DEMANDS\s*\((.*?)\n\)", re.DOTALL)
        demands_match = demands_pattern.search(data)
        demands_content = demands_match.group(1).strip() if demands_match else ''

        demands = re.findall(r"\(\s+([a-zA-Z0-9\.]+)\s+([a-zA-Z0-9\.]+)\s+\)\s+\d+\s+([\d.]+)\s+", demands_content)
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
        row = {link_name: node_traffic[link_name] for link_name in link_names}
        rows.append(row)
        times.append(_time)

    # Create a DataFrame where each row corresponds to a file
    traffic_df = pd.DataFrame(rows, index=times)
    traffic_df.index.name = "Datetime"
    print(traffic_df)
    traffic_df.to_csv(f"data/{network_name}_traffic.csv")


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

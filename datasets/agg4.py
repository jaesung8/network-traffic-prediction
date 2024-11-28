# -*- coding: utf-8 -*-
import re
import codecs
import glob
import os

# 경로 데이터를 딕셔너리로 정의합니다.
path_dict = {
    ('ATLAM5', 'ATLAng'): ['ATLAM5', 'ATLAng'],
    ('ATLAM5', 'CHINng'): ['ATLAM5', 'ATLAng', 'IPLSng', 'CHINng'],
    ('ATLAM5', 'DNVRng'): ['ATLAM5', 'ATLAng', 'IPLSng', 'KSCYng', 'DNVRng'],
    ('ATLAM5', 'HSTNng'): ['ATLAM5', 'ATLAng', 'HSTNng'],
    ('ATLAM5', 'IPLSng'): ['ATLAM5', 'ATLAng', 'IPLSng'],
    ('ATLAM5', 'KSCYng'): ['ATLAM5', 'ATLAng', 'IPLSng', 'KSCYng'],
    ('ATLAM5', 'LOSAng'): ['ATLAM5', 'ATLAng', 'HSTNng', 'LOSAng'],
    ('ATLAM5', 'NYCMng'): ['ATLAM5', 'ATLAng', 'WASHng', 'NYCMng'],
    ('ATLAM5', 'SNVAng'): ['ATLAM5', 'ATLAng', 'IPLSng', 'KSCYng', 'DNVRng', 'SNVAng'],
    ('ATLAM5', 'STTLng'): ['ATLAM5', 'ATLAng', 'IPLSng', 'KSCYng', 'DNVRng', 'STTLng'],
    ('ATLAM5', 'WASHng'): ['ATLAM5', 'ATLAng', 'WASHng'],
    ('ATLAng', 'ATLAM5'): ['ATLAng', 'ATLAM5'],
    ('ATLAng', 'CHINng'): ['ATLAng', 'IPLSng', 'CHINng'],
    ('ATLAng', 'DNVRng'): ['ATLAng', 'IPLSng', 'KSCYng', 'DNVRng'],
    ('ATLAng', 'HSTNng'): ['ATLAng', 'HSTNng'],
    ('ATLAng', 'IPLSng'): ['ATLAng', 'IPLSng'],
    ('ATLAng', 'KSCYng'): ['ATLAng', 'IPLSng', 'KSCYng'],
    ('ATLAng', 'LOSAng'): ['ATLAng', 'HSTNng', 'LOSAng'],
    ('ATLAng', 'NYCMng'): ['ATLAng', 'WASHng', 'NYCMng'],
    ('ATLAng', 'SNVAng'): ['ATLAng', 'IPLSng', 'KSCYng', 'DNVRng', 'SNVAng'],
    ('ATLAng', 'STTLng'): ['ATLAng', 'IPLSng', 'KSCYng', 'DNVRng', 'STTLng'],
    ('ATLAng', 'WASHng'): ['ATLAng', 'WASHng'],
    ('CHINng', 'ATLAng'): ['CHINng', 'IPLSng', 'ATLAng'],
    ('CHINng', 'ATLAM5'): ['CHINng', 'IPLSng', 'ATLAng', 'ATLAM5'],
    ('CHINng', 'DNVRng'): ['CHINng', 'IPLSng', 'KSCYng', 'DNVRng'],
    ('CHINng', 'HSTNng'): ['CHINng', 'IPLSng', 'ATLAng', 'HSTNng'],
    ('CHINng', 'IPLSng'): ['CHINng', 'IPLSng'],
    ('CHINng', 'KSCYng'): ['CHINng', 'IPLSng', 'KSCYng'],
    ('CHINng', 'LOSAng'): ['CHINng', 'IPLSng', 'KSCYng', 'DNVRng', 'SNVAng', 'LOSAng'],
    ('CHINng', 'NYCMng'): ['CHINng', 'NYCMng'],
    ('CHINng', 'SNVAng'): ['CHINng', 'IPLSng', 'KSCYng', 'DNVRng', 'SNVAng'],
    ('CHINng', 'STTLng'): ['CHINng', 'IPLSng', 'KSCYng', 'DNVRng', 'STTLng'],
    ('CHINng', 'WASHng'): ['CHINng', 'NYCMng', 'WASHng'],
    ('DNVRng', 'CHINng'): ['DNVRng', 'KSCYng', 'IPLSng', 'CHINng'],
    ('DNVRng', 'ATLAng'): ['DNVRng', 'KSCYng', 'IPLSng', 'ATLAng'],
    ('DNVRng', 'ATLAM5'): ['DNVRng', 'KSCYng', 'IPLSng', 'ATLAng', 'ATLAM5'],
    ('DNVRng', 'HSTNng'): ['DNVRng', 'KSCYng', 'HSTNng'],
    ('DNVRng', 'IPLSng'): ['DNVRng', 'KSCYng', 'IPLSng'],
    ('DNVRng', 'KSCYng'): ['DNVRng', 'KSCYng'],
    ('DNVRng', 'LOSAng'): ['DNVRng', 'SNVAng', 'LOSAng'],
    ('DNVRng', 'NYCMng'): ['DNVRng', 'KSCYng', 'IPLSng', 'CHINng', 'NYCMng'],
    ('DNVRng', 'SNVAng'): ['DNVRng', 'SNVAng'],
    ('DNVRng', 'STTLng'): ['DNVRng', 'STTLng'],
    ('DNVRng', 'WASHng'): ['DNVRng', 'KSCYng', 'IPLSng', 'CHINng', 'NYCMng', 'WASHng'],
    ('HSTNng', 'DNVRng'): ['HSTNng', 'KSCYng', 'DNVRng'],
    ('HSTNng', 'CHINng'): ['HSTNng', 'ATLAng', 'IPLSng', 'CHINng'],
    ('HSTNng', 'ATLAng'): ['HSTNng', 'ATLAng'],
    ('HSTNng', 'ATLAM5'): ['HSTNng', 'ATLAng', 'ATLAM5'],
    ('HSTNng', 'IPLSng'): ['HSTNng', 'ATLAng', 'IPLSng'],
    ('HSTNng', 'KSCYng'): ['HSTNng', 'KSCYng'],
    ('HSTNng', 'LOSAng'): ['HSTNng', 'LOSAng'],
    ('HSTNng', 'NYCMng'): ['HSTNng', 'ATLAng', 'WASHng', 'NYCMng'],
    ('HSTNng', 'SNVAng'): ['HSTNng', 'LOSAng', 'SNVAng'],
    ('HSTNng', 'STTLng'): ['HSTNng', 'KSCYng', 'DNVRng', 'STTLng'],
    ('HSTNng', 'WASHng'): ['HSTNng', 'ATLAng', 'WASHng'],
    ('IPLSng', 'HSTNng'): ['IPLSng', 'ATLAng', 'HSTNng'],
    ('IPLSng', 'DNVRng'): ['IPLSng', 'KSCYng', 'DNVRng'],
    ('IPLSng', 'CHINng'): ['IPLSng', 'CHINng'],
    ('IPLSng', 'ATLAng'): ['IPLSng', 'ATLAng'],
    ('IPLSng', 'ATLAM5'): ['IPLSng', 'ATLAng', 'ATLAM5'],
    ('IPLSng', 'KSCYng'): ['IPLSng', 'KSCYng'],
    ('IPLSng', 'LOSAng'): ['IPLSng', 'KSCYng', 'DNVRng', 'SNVAng', 'LOSAng'],
    ('IPLSng', 'NYCMng'): ['IPLSng', 'CHINng', 'NYCMng'],
    ('IPLSng', 'SNVAng'): ['IPLSng', 'KSCYng', 'DNVRng', 'SNVAng'],
    ('IPLSng', 'STTLng'): ['IPLSng', 'KSCYng', 'DNVRng', 'STTLng'],
    ('IPLSng', 'WASHng'): ['IPLSng', 'ATLAng', 'WASHng'],
    ('KSCYng', 'IPLSng'): ['KSCYng', 'IPLSng'],
    ('KSCYng', 'HSTNng'): ['KSCYng', 'HSTNng'],
    ('KSCYng', 'DNVRng'): ['KSCYng', 'DNVRng'],
    ('KSCYng', 'CHINng'): ['KSCYng', 'IPLSng', 'CHINng'],
    ('KSCYng', 'ATLAng'): ['KSCYng', 'IPLSng', 'ATLAng'],
    ('KSCYng', 'ATLAM5'): ['KSCYng', 'IPLSng', 'ATLAng', 'ATLAM5'],
    ('KSCYng', 'LOSAng'): ['KSCYng', 'DNVRng', 'SNVAng', 'LOSAng'],
    ('KSCYng', 'NYCMng'): ['KSCYng', 'IPLSng', 'CHINng', 'NYCMng'],
    ('KSCYng', 'SNVAng'): ['KSCYng', 'DNVRng', 'SNVAng'],
    ('KSCYng', 'STTLng'): ['KSCYng', 'DNVRng', 'STTLng'],
    ('KSCYng', 'WASHng'): ['KSCYng', 'IPLSng', 'ATLAng', 'WASHng'],
    ('LOSAng', 'KSCYng'): ['LOSAng', 'SNVAng', 'DNVRng', 'KSCYng'],
    ('LOSAng', 'IPLSng'): ['LOSAng', 'SNVAng', 'DNVRng', 'KSCYng', 'IPLSng'],
    ('LOSAng', 'HSTNng'): ['LOSAng', 'HSTNng'],
    ('LOSAng', 'DNVRng'): ['LOSAng', 'SNVAng', 'DNVRng'],
    ('LOSAng', 'CHINng'): ['LOSAng', 'SNVAng', 'DNVRng', 'KSCYng', 'IPLSng', 'CHINng'],
    ('LOSAng', 'ATLAng'): ['LOSAng', 'HSTNng', 'ATLAng'],
    ('LOSAng', 'ATLAM5'): ['LOSAng', 'HSTNng', 'ATLAng', 'ATLAM5'],
    ('LOSAng', 'NYCMng'): ['LOSAng', 'HSTNng', 'ATLAng', 'WASHng', 'NYCMng'],
    ('LOSAng', 'SNVAng'): ['LOSAng', 'SNVAng'],
    ('LOSAng', 'STTLng'): ['LOSAng', 'SNVAng', 'STTLng'],
    ('LOSAng', 'KSCYng'): ['LOSAng', 'DNVRng', 'KSCYng'],
    ('LOSAng', 'IPLSng'): ['LOSAng', 'DNVRng', 'KSCYng', 'IPLSng'],
    ('LOSAng', 'HSTNng'): ['LOSAng', 'DNVRng', 'KSCYng', 'HSTNng'],
    ('LOSAng', 'DNVRng'): ['LOSAng', 'DNVRng'],
    ('LOSAng', 'CHINng'): ['LOSAng', 'DNVRng', 'KSCYng', 'IPLSng', 'CHINng'],
    ('LOSAng', 'ATLAng'): ['LOSAng', 'DNVRng', 'KSCYng', 'IPLSng', 'ATLAng'],
    ('LOSAng', 'ATLAM5'): ['LOSAng', 'DNVRng', 'KSCYng', 'IPLSng', 'ATLAng', 'ATLAM5'],
    ('LOSAng', 'NYCMng'): ['LOSAng', 'HSTNng', 'ATLAng', 'WASHng', 'NYCMng'],
    ('LOSAng', 'SNVAng'): ['LOSAng', 'SNVAng'],
    ('LOSAng', 'STTLng'): ['LOSAng', 'SNVAng', 'STTLng'],
    ('LOSAng', 'WASHng'): ['LOSAng', 'DNVRng', 'KSCYng', 'IPLSng', 'ATLAng', 'WASHng'],
    ('NYCMng', 'LOSAng'): ['NYCMng', 'WASHng', 'ATLAng', 'HSTNng', 'LOSAng'],
    ('NYCMng', 'KSCYng'): ['NYCMng', 'CHINng', 'IPLSng', 'KSCYng'],
    ('NYCMng', 'IPLSng'): ['NYCMng', 'CHINng', 'IPLSng'],
    ('NYCMng', 'HSTNng'): ['NYCMng', 'WASHng', 'ATLAng', 'HSTNng'],
    ('NYCMng', 'DNVRng'): ['NYCMng', 'CHINng', 'IPLSng', 'KSCYng', 'DNVRng'],
    ('NYCMng', 'CHINng'): ['NYCMng', 'CHINng'],
    ('NYCMng', 'ATLAng'): ['NYCMng', 'WASHng', 'ATLAng'],
    ('NYCMng', 'ATLAM5'): ['NYCMng', 'WASHng', 'ATLAng', 'ATLAM5'],
    ('NYCMng', 'SNVAng'): ['NYCMng', 'CHINng', 'IPLSng', 'KSCYng', 'DNVRng', 'SNVAng'],
    ('NYCMng', 'STTLng'): ['NYCMng', 'CHINng', 'IPLSng', 'KSCYng', 'DNVRng', 'STTLng'],
    ('NYCMng', 'WASHng'): ['NYCMng', 'WASHng'],
    ('SNVAng', 'NYCMng'): ['SNVAng', 'DNVRng', 'KSCYng', 'IPLSng', 'CHINng', 'NYCMng'],
    ('SNVAng', 'LOSAng'): ['SNVAng', 'LOSAng'],
    ('SNVAng', 'KSCYng'): ['SNVAng', 'DNVRng', 'KSCYng'],
    ('SNVAng', 'IPLSng'): ['SNVAng', 'DNVRng', 'KSCYng', 'IPLSng'],
    ('SNVAng', 'HSTNng'): ['SNVAng', 'LOSAng', 'HSTNng'],
    ('SNVAng', 'DNVRng'): ['SNVAng', 'DNVRng'],
    ('SNVAng', 'CHINng'): ['SNVAng', 'DNVRng', 'KSCYng', 'IPLSng', 'CHINng'],
    ('SNVAng', 'ATLAng'): ['SNVAng', 'DNVRng', 'KSCYng', 'IPLSng', 'ATLAng'],
    ('SNVAng', 'ATLAM5'): ['SNVAng', 'DNVRng', 'KSCYng', 'IPLSng', 'ATLAng', 'ATLAM5'],
    ('SNVAng', 'STTLng'): ['SNVAng', 'STTLng'],
    ('SNVAng', 'WASHng'): ['SNVAng', 'LOSAng', 'HSTNng', 'ATLAng', 'WASHng'],
    ('STTLng', 'SNVAng'): ['STTLng', 'SNVAng'],
    ('STTLng', 'NYCMng'): ['STTLng', 'DNVRng', 'KSCYng', 'IPLSng', 'CHINng', 'NYCMng'],
    ('STTLng', 'LOSAng'): ['STTLng', 'SNVAng', 'LOSAng'],
    ('STTLng', 'KSCYng'): ['STTLng', 'DNVRng', 'KSCYng'],
    ('STTLng', 'IPLSng'): ['STTLng', 'DNVRng', 'KSCYng', 'IPLSng'],
    ('STTLng', 'HSTNng'): ['STTLng', 'DNVRng', 'KSCYng', 'HSTNng'],
    ('STTLng', 'DNVRng'): ['STTLng', 'DNVRng'],
    ('STTLng', 'CHINng'): ['STTLng', 'DNVRng', 'KSCYng', 'IPLSng', 'CHINng'],
    ('STTLng', 'ATLAng'): ['STTLng', 'DNVRng', 'KSCYng', 'IPLSng', 'ATLAng'],
    ('STTLng', 'ATLAM5'): ['STTLng', 'DNVRng', 'KSCYng', 'IPLSng', 'ATLAng', 'ATLAM5'],
    ('STTLng', 'WASHng'): ['STTLng', 'DNVRng', 'KSCYng', 'IPLSng', 'ATLAng', 'WASHng'],
    ('WASHng', 'SNVAng'): ['WASHng', 'ATLAng', 'HSTNng', 'LOSAng', 'SNVAng'],
    ('WASHng', 'NYCMng'): ['WASHng', 'NYCMng'],
    ('WASHng', 'LOSAng'): ['WASHng', 'ATLAng', 'HSTNng', 'LOSAng'],
    ('WASHng', 'KSCYng'): ['WASHng', 'ATLAng', 'IPLSng', 'KSCYng'],
    ('WASHng', 'IPLSng'): ['WASHng', 'ATLAng', 'IPLSng'],
    ('WASHng', 'HSTNng'): ['WASHng', 'ATLAng', 'HSTNng'],
    ('WASHng', 'DNVRng'): ['WASHng', 'NYCMng', 'CHINng', 'IPLSng', 'KSCYng', 'DNVRng'],
    ('WASHng', 'CHINng'): ['WASHng', 'NYCMng', 'CHINng'],
    ('WASHng', 'ATLAng'): ['WASHng', 'ATLAng'],
    ('WASHng', 'ATLAM5'): ['WASHng', 'ATLAng', 'ATLAM5'],
    ('WASHng', 'STTLng'): ['WASHng', 'ATLAng', 'IPLSng', 'KSCYng', 'DNVRng', 'STTLng'],
}


def parse_sndlib_demands(file_path):
    demands_section = False
    demands = []
    with codecs.open(file_path, 'r', 'utf-8') as file:
        for line in file:
            line = line.strip()
            # Ignore comments and empty lines
            if not line or line.startswith('#'):
                continue
            # Check for the start of the DEMANDS section
            if re.match(r'DEMANDS\s*\(', line):
                demands_section = True
                continue
            # Check for the end of the DEMANDS section
            if demands_section and re.match(r'\)', line):
                demands_section = False
                continue
            # Parse data within the DEMANDS section
            if demands_section:
                # Remove trailing commas or comments
                line = re.sub(r',?\s*(#.*)?$', '', line)
                # Skip empty lines
                if not line:
                    continue
                # Use regex to extract data
                match = re.match(r'\S+\s*\(\s*(\S+)\s+(\S+)\s*\)\s+\S+\s+([\d\.E+-]+)', line)
                if match:
                    source = match.group(1)
                    target = match.group(2)
                    demand_value = float(match.group(3))
                    demands.append((source, target, demand_value))
                else:
                    print("No match for line:", line)
    return demands

# Get the list of files
file_pattern = 'demandMatrix-abilene-zhang-5min-200409*-*.txt'  # Updated pattern to match all March dates and times
file_list = glob.glob(file_pattern)
# Sort the file list
file_list.sort()

# Initialize a dictionary to store total demand per node across all files
total_node_demands = {}
a = 0
# Open the output file
with codecs.open('aaa.txt', 'w', 'utf-8') as output_file:
    # Write header

    
    # Process each file
    for file_path in file_list: 
        if a > 288:
           a = 0   
        # 파일 이름에서 날짜를 추출합니다.
        filename = os.path.basename(file_path)
        # 정규 표현식을 사용하여 'zhang-5min-YYYYMMDD-' 패턴에서 YYYYMMDD를 추출합니다.
        match = re.search(r'zhang-5min-(\d{8})-', filename)
        if match:
            date_str = match.group(1)
        else:
            date_str = "UnknownDate"
            print(f"Date not found in filename: {filename}")
        
        # DEMANDS 데이터를 파싱합니다.
        demands = parse_sndlib_demands(file_path)
        
        # 각 수요 항목에 대해 경로를 따라 수요 값을 누적합니다.
        for source, target, demand_value in demands:
            # (SRC, DST) 쌍에 해당하는 경로를 가져옵니다.
            path = path_dict.get((source, target))
            if not path:
                print(f"Path not found for SRC: {source}, DST: {target}")
                continue  # 경로가 없으면 다음 수요 항목으로 넘어갑니다.
            # 경로에 있는 각 노드에 수요 값을 누적합니다.
            for node in path:
                if node in total_node_demands:
                    total_node_demands[node] += demand_value
                else:
                    total_node_demands[node] = demand_value
        for node, total_demand in sorted(total_node_demands.items()):
            output_file.write(f"{total_demand} ")   
        if a == 0 :
            output_file.write(f"  {date_str} ")    
            a=a+1       
        output_file.write(f"\n")     
        total_node_demands = {}
        a = a+1

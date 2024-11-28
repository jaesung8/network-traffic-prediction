
import re
import codecs
import glob
import os 

def parse_sndlib_demands(file_path):
    demands_section = False
    demands = []
    with codecs.open(file_path, 'r', 'utf-8') as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            if re.match(r'DEMANDS\s*\(', line):
                demands_section = True
                continue

            if demands_section and re.match(r'\)', line):
                demands_section = False
                continue

            if demands_section:

                line = re.sub(r',?\s*(#.*)?$', '', line)

                if not line:
                    continue

                match = re.match(r'\S+\s*\(\s*(\S+)\s+(\S+)\s*\)\s+\S+\s+([\d\.E+-]+)', line)
                if match:
                    source = match.group(1)
                    target = match.group(2)
                    demand_value = float(match.group(3))
                    demands.append((source, demand_value))
                    demands.append((target, demand_value))
                else:
                    print("No match for line:", line)
    return demands



file_pattern = 'demandMatrix-abilene-zhang-5min-200409*-*.txt'  
file_list = glob.glob(file_pattern)

file_list.sort()
a = 0

with codecs.open('out.txt', 'w', 'utf-8') as output_file:
    
    for file_path in file_list: 
        if a > 288:
           a = 0        

        filename = os.path.basename(file_path)
       
        match = re.search(r'zhang-5min-(\d{8})-', filename)
        if match:
            date_str = match.group(1)
        else:
            date_str = "UnknownDate"
        
        
        demands = parse_sndlib_demands(file_path)

        node_demands = {}
        for node, demand_value in demands:
            if node in node_demands:
                node_demands[node] += demand_value
            else:
                node_demands[node] = demand_value

        
        for node, total_demand in node_demands.items():
            output_file.write('{} '.format(total_demand))
        if a == 0 :
           output_file.write(f"  {date_str} ")    
           a=a+1          
        output_file.write('\n')
        a = a+1


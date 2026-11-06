import os
import networkx as nx
import pandas as pd

def convert_gexf_to_csv_and_graphml(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".gexf"):
            file_path = os.path.join(input_folder, filename)
            graph = nx.read_gexf(file_path)
            
            # Convert to CSV
            csv_filename = filename.replace(".gexf", ".csv")
            csv_path = os.path.join(output_folder, csv_filename)
            edges = nx.to_pandas_edgelist(graph)
            edges['source'] = 'a' + edges['source'].astype(str)
            edges['target'] = 'a' + edges['target'].astype(str)
            edges = edges[['source', 'target']]
            edges.to_csv(csv_path, header=None,index=False)
            
            # Convert to GraphML
            graphml_filename = filename.replace(".gexf", ".graphml")
            
            graphml_path = os.path.join(output_folder, graphml_filename)
            nx.write_graphml(graph, graphml_path)
            
            print(f"Converted {filename} to {csv_filename} and {graphml_filename}")

input_folder = "/home/cau/mydata/LLM_OD_llama.3.1/data/min03"
output_folder = "/home/cau/mydata/LLM_OD_llama.3.1/data/min03"
convert_gexf_to_csv_and_graphml(input_folder, output_folder)
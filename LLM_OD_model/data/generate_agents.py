import json
import os
import networkx as nx

def generate_agents(n=100, opinions=7):
    res = []

    b = int(n / opinions)
    current = 0
    op = 0
    for i in range(n):
        agent = {"name": f"a{i}", "status": op}
        current = current + 1

        if current >= b:
            op += 1
            current = 0

        res.append(agent)

    return res


def generate_unbalanced_agents(g=nx.Graph(),):
    res = []

    for i in g.nodes(data=True):
        agent = {"name": f"a{i[0]}"}

        if i[1]['class'] == 'blue':
            agent['status'] = 6
            agent["llm_name"] = "meta-llama/Llama-3.2-3B-Instruct"
        else:
            agent['status'] = 0
            agent["llm_name"] = "meta-llama/Llama-3.2-3B-Instruct"

        res.append(agent)

    return res

if __name__ == "__main__":
    print("working")
    for min in ["0.1", "0.3", "0.5"]:
        for h_val in ['0.0', '0.5', '0.25', '0.75', '1.0']:
            g = nx.read_graphml(f"./graphs/PAH-min{min}-h{h_val}.graphml")
            agents = generate_unbalanced_agents(g=g)
            json.dump(agents, open(f"../sample_data/reverse_agents_PAH-min{min}-h{h_val}_100_meta-llama_Llama-3.2-3B-Instruct.json", "w"))


    


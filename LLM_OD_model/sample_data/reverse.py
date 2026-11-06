import json

for l in ["A", "B", "C", "D", "E"]:
    data = json.load(open(f"agents_peel-{l}_40_llm_llama3,mistral.json"))
    users = []
    for u in data:
        if u['llm_name'] == 'mistral':
            u['llm_name'] = 'llama3'
        else:
            u['llm_name'] = 'mistral'

        users.append(u)
    json.dump(users, open(f"agents_peel-{l}_40_llm_mistral,llama3.json", "w"))
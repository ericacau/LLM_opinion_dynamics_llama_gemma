from llm_network.simulator import LLMOpinionSimulator
import llm_network as llmn
import networkx as nx
import json
import sys
import os
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = MODEL_ROOT.parent
import logging

logging.getLogger("autogen.oai.client").setLevel(logging.ERROR)
import copy
import warnings
from autogen import ConversableAgent


def model_dump(model_obj):
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    elif hasattr(model_obj, "dict"):
        return model_obj.dict()
    return model_obj


# Monkeypatch autogen ConversableAgent to force strict role alternation (user/assistant)
# starting with 'user' for strict OpenAI-compatible API servers (like vLLM).
original_generate_oai_reply_from_client = ConversableAgent._generate_oai_reply_from_client

def patched_generate_oai_reply_from_client(self, llm_client, messages, cache):
    copied_messages = copy.deepcopy(messages)
    all_messages = []
    for message in copied_messages:
        tool_responses = message.get("tool_responses", [])
        if tool_responses:
            all_messages += tool_responses
            if message.get("role") != "tool":
                all_messages.append({key: message[key] for key in message if key != "tool_responses"})
        else:
            all_messages.append(message)
            
    system_msgs = [m for m in all_messages if m.get("role") == "system"]
    other_msgs = [m for m in all_messages if m.get("role") != "system"]
    
    if other_msgs:
        # Force first non-system message to be 'user'
        if other_msgs[0].get("role") == "assistant":
            other_msgs[0]["role"] = "user"
            
        # Merge consecutive messages with identical roles
        alternated_msgs = []
        last_role = None
        for m in other_msgs:
            role = m.get("role")
            if role == last_role:
                if alternated_msgs:
                    alternated_msgs[-1]["content"] = (
                        str(alternated_msgs[-1].get("content", "")) + "\n\n" + str(m.get("content", ""))
                    )
            else:
                alternated_msgs.append(m)
                last_role = role
                
        # Force last message to be 'user' (cannot end with 'assistant' when querying API)
        if alternated_msgs and alternated_msgs[-1].get("role") == "assistant":
            alternated_msgs[-1]["role"] = "user"
            
        all_messages = system_msgs + alternated_msgs

    # Directly execute the client request with cleaned messages list
    response = llm_client.create(
        context=all_messages[-1].pop("context", None) if all_messages else None,
        messages=all_messages,
        cache=cache,
    )
    extracted_response = llm_client.extract_text_or_completion_object(response)[0]

    if extracted_response is None:
        warnings.warn(f"Extracted_response from {response} is None.", UserWarning)
        return None
    if not isinstance(extracted_response, str) and hasattr(extracted_response, "model_dump"):
        extracted_response = model_dump(extracted_response)
    if isinstance(extracted_response, dict):
        if extracted_response.get("function_call"):
            extracted_response["function_call"]["name"] = self._normalize_name(
                extracted_response["function_call"]["name"]
            )
        for tool_call in extracted_response.get("tool_calls") or []:
            tool_call["function"]["name"] = self._normalize_name(tool_call["function"]["name"])
            if tool_call.get("id") is None:
                tool_call.pop("id")
            if tool_call.get("type") is None:
                tool_call.pop("type")
    return extracted_response

ConversableAgent._generate_oai_reply_from_client = patched_generate_oai_reply_from_client


def execute(
    models,
    config_list,
    network,
    n,
    name,
    theme=None,
    prompt_type="theseus_opinion_distr",
    experiment="unbalanced",
    n_agents=100,
    folder='',
    use_vllm=False,
    vllm_url=None,
    vllm_mode="server",
    n_iterations=100,
    monitor_type="MonitorOpinionDistribution",
):
    llm_config = {
        "config_list": None,
        "seed": 42,
        "max_tokens": 8192,
        "temperature": 0.8,
    }

    # Create a network of agents from files
    net = llmn.Network()
    if network is not None:
        network_name = os.path.basename(network.strip())
        min_part = network_name.split('-')[1]
    else:
        min_part = experiment.split('-')[1]
    data_folder = f"min{min_part.split('min')[1].replace('.', '')}"
    model_name = models.replace(":", "")
    agent_file = MODEL_ROOT / "sample_data" / f"agents_{experiment}_{n_agents}_{model_name}.json"
    print(agent_file)
    net.add_agents(str(agent_file))

    if network is not None:
        network = network.strip()
        network_path = MODEL_ROOT / "data" / data_folder / network
        g = nx.read_edgelist(network_path, nodetype=str, delimiter=",")
        g = nx.relabel_nodes(g, lambda x: x.strip())
        net.set_network(g)

    # Load instructions and opinion map
    instructions_path = MODEL_ROOT / "sample_data" / f"agents_instructions_{prompt_type}.json"
    instructions = json.load(open(instructions_path))
    opinion_map = json.load(open(MODEL_ROOT / "sample_data" / "opinion_map.json"))

    if use_vllm:
        monitor_map = {
            "Monitor": "MonitorVLLM",
            "MonitorOpinionDistribution": "MonitorOpinionDistributionVLLM"
        }
        monitor_type = monitor_map.get(monitor_type, monitor_type + "VLLM")

    # run the simulation
    sim = LLMOpinionSimulator(
        llm_config,
        config_list,
        verbose=False,
        save_agents_debates=True,
        monitor_type=monitor_type, 
        agents_instruction=instructions,
        opinion_map=opinion_map,
        min_opinion=0,
        max_opinion=6,
        vllm_url=vllm_url,
        vllm_mode=vllm_mode,
    )
    sim.set_agents(net)
    
    output_dir = PACKAGE_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    sim.run(
        n_iterations=n_iterations,
        themes=theme,
        output_file=str(output_dir / f"{prompt_type}_{name.split('.')[0]}_{model_name}_{n}_{experiment}.jsonl"),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM Opinion Simulator Execution script")
    parser.add_argument("models", type=str, help="Comma-separated list of models to use")
    parser.add_argument("run_n", type=int, help="Number of runs")
    parser.add_argument("theme_name", type=str, help="Theme JSON file name")
    parser.add_argument("exp_name", type=str, help="Experiment name")
    parser.add_argument("n_agents", type=int, help="Number of agents")
    parser.add_argument("network", type=str, nargs="?", default=None, help="Optional network file path")
    parser.add_argument("--prompt-type", type=str, default="theseus_opinion_distr", choices=["theseus", "theseus_opinion_distr"], help="Prompt/instructions type to use")
    parser.add_argument("--vllm", action="store_true", help="Use batched vLLM backend")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1", help="vLLM server API base URL")
    parser.add_argument("--vllm-mode", type=str, default="server", choices=["server", "offline"], help="vLLM batching mode")
    parser.add_argument("-i", "--iterations", type=int, default=100, help="Number of simulation iterations")
    parser.add_argument("--monitor", type=str, default="MonitorOpinionDistribution", choices=["Monitor", "MonitorOpinionDistribution"], help="Monitor type to use")

    args = parser.parse_args()

    models = args.models
    run_n = args.run_n
    theme_name = args.theme_name
    exp_name = args.exp_name
    n_agents = args.n_agents
    network = args.network

    model_list = models.split(",")
    config_list = {}

    # Create a configuration for each model
    for model in model_list:
        config_list[model] = {
            "model": f"{model}",
            "base_url": args.vllm_url if args.vllm_url else "http://localhost:11434/v1",
            "api_type": "openai",
            "api_key": "NULL",
        }

    #identify current path for different minority classes
    min_part = exp_name.split('-')[1] 
    folder = f"min{min_part.split('min')[1].replace('.', '')}"  

    theme = json.load(open(MODEL_ROOT / "themes" / theme_name, "r"))

    for n in range(run_n):
        execute(
            models=models,
            config_list=config_list,
            network=network,
            n=n,
            name=theme_name,
            theme=theme,
            prompt_type=args.prompt_type,
            experiment=exp_name,
            n_agents=n_agents,
            folder=folder,
            use_vllm=args.vllm,
            vllm_url=args.vllm_url,
            vllm_mode=args.vllm_mode,
            n_iterations=args.iterations,
            monitor_type=args.monitor,
        )

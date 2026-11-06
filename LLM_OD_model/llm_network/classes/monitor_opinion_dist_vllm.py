import asyncio
import collections
import tqdm
from openai import AsyncOpenAI
from .monitor_opinion_dist import MonitorOpinionDistribution


# Cache offline engine globally to avoid reinitialization overhead
_offline_engines = {}

def get_offline_engine(model_name):
    if model_name not in _offline_engines:
        try:
            from vllm import LLM
            print(f"Initializing offline vLLM engine for model: {model_name}")
            _offline_engines[model_name] = LLM(model=model_name, trust_remote_code=True)
        except ImportError:
            raise ImportError(
                "The 'vllm' package is required for offline mode but is not installed. "
                "Please run pip install vllm or use --vllm-mode server."
            )
    return _offline_engines[model_name]


def format_messages_for_llm(messages, model_name, tokenizer=None):
    """
    Format chat messages into a single prompt string for offline vLLM inference.
    Uses apply_chat_template if available, otherwise falls back to Llama-3/Mistral heuristics.
    """
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    # Extract system prompt if present
    system_content = ""
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]

    prompt = ""
    if "llama3" in model_name.lower():
        if system_content:
            prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_content}<|eot_id|>"
        else:
            prompt += "<|begin_of_text|>"
        for msg in messages:
            if msg["role"] == "system":
                continue
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        # Mistral / Llama 2 style fallback
        if system_content:
            prompt += f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n"
        else:
            prompt += "<s>[INST] "
        
        first_user = True
        for msg in messages:
            if msg["role"] == "system":
                continue
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                if not first_user:
                    prompt += " </s><s>[INST] "
                prompt += content + " [/INST]"
                first_user = False
            elif role == "assistant":
                prompt += " " + content
    return prompt


async def generate_server_batch_async(model_requests, default_url="http://localhost:8000/v1"):
    """
    Generate responses in parallel from a running OpenAI-compatible vLLM/Ollama API server.
    """
    sem = asyncio.Semaphore(50)  # Concurrency limit
    clients = {}

    def get_client(base_url, api_key):
        key = (base_url, api_key)
        if key not in clients:
            clients[key] = AsyncOpenAI(base_url=base_url, api_key=api_key)
        return clients[key]

    async def call_api(item):
        model = item["model"]
        base_url = item.get("base_url") or default_url
        api_key = item.get("api_key") or "NULL"
        messages = item["messages"]
        client = get_client(base_url, api_key)

        async with sem:
            for attempt in range(3):
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.8,
                        max_tokens=2048,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"API attempt {attempt+1} failed for model {model}: {e}")
                    await asyncio.sleep(1)
            return ""

    tasks = [call_api(item) for item in model_requests]
    results = await asyncio.gather(*tasks)
    
    # Close all created clients to prevent event loop closure errors
    for client in clients.values():
        try:
            await client.close()
        except Exception:
            pass
            
    return results


def generate_server_batch(model_requests, default_url="http://localhost:8000/v1"):
    return asyncio.run(generate_server_batch_async(model_requests, default_url))


def generate_offline_batch(model_requests):
    """
    Generate responses in parallel using the offline vLLM library engine.
    """
    from vllm import SamplingParams

    grouped = collections.defaultdict(list)
    for idx, (model_name, messages) in enumerate(model_requests):
        grouped[model_name].append((idx, messages))

    results = [None] * len(model_requests)

    for model_name, items in grouped.items():
        engine = get_offline_engine(model_name)
        prompts = []
        for _, messages in items:
            formatted = format_messages_for_llm(messages, model_name, engine.get_tokenizer())
            prompts.append(formatted)

        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=2048,
        )
        outputs = engine.generate(prompts, sampling_params)
        for (original_idx, _), out in zip(items, outputs):
            results[original_idx] = out.outputs[0].text

    return results


class MonitorOpinionDistributionVLLM(MonitorOpinionDistribution):
    def __init__(self, *args, **kwargs):
        self.vllm_url = kwargs.pop("vllm_url", None)
        self.vllm_mode = kwargs.pop("vllm_mode", "server")
        super().__init__(*args, **kwargs)

    def generate_batch(self, requests):
        """
        Routes the requests to either the server-based or offline vLLM backend.
        """
        if self.vllm_mode == "offline":
            offline_requests = [(req["model"], req["messages"]) for req in requests]
            return generate_offline_batch(offline_requests)
        else:
            default_url = self.vllm_url if self.vllm_url else "http://localhost:8000/v1"
            return generate_server_batch(requests, default_url)

    def debate_batch(self, pairs, theme):
        """
        Runs the debates for a batch of agent pairs concurrently in rounds.
        """
        # Round 1: Opponent statement
        reqs_round1 = []
        instructions = []  # save for later rounds
        
        for discussant, opponent in pairs:
            discussant_opinion = self.args["opinion_map"][str(self.statuses[discussant.name])]
            opponent_opinion = self.args["opinion_map"][str(self.statuses[opponent.name])]

            neighbors = discussant.get_neighbors()
            opinion_distribution = {k: 0 for k in self.args["opinion_map"].values()}
            for n in neighbors:
                opinion_distribution[self.args["opinion_map"][str(self.statuses[n.name])]] += 1
            opinion_distribution = {k: v / len(neighbors) for k, v in opinion_distribution.items()}
            opinion_distribution = "\n".join([f"{k}: {v:.1%}" for k, v in opinion_distribution.items()])

            u1_instruction = self.agents_instruction["discussant"].format(**locals())
            u2_instruction = self.agents_instruction["opponent"].format(**locals())
            
            # Save formatted instructions for the pair
            instructions.append((u1_instruction, u2_instruction))

            # Retrieve backend details
            opponent_llm_name = opponent.get_llm_name()
            opponent_config = self.config_list[0][opponent_llm_name]
            
            messages_r1 = [
                {"role": "system", "content": u2_instruction},
                {"role": "user", "content": f' What do you think of the following statement?: "{theme}" '}
            ]
            
            req_item = {
                "model": opponent_config.get("model", opponent_llm_name),
                "messages": messages_r1,
            }
            if not self.vllm_url and "base_url" in opponent_config:
                req_item["base_url"] = opponent_config["base_url"]
            if "api_key" in opponent_config:
                req_item["api_key"] = opponent_config["api_key"]
                
            reqs_round1.append(req_item)

        print(f"Generating Round 1 (Opponent statement) for {len(pairs)} pairs...")
        text1_opponent_list = self.generate_batch(reqs_round1)

        # Round 2: Discussant answer
        reqs_round2 = []
        for i, (discussant, opponent) in enumerate(pairs):
            u1_instruction, _ = instructions[i]
            text1_opponent = text1_opponent_list[i]
            
            discussant_llm_name = discussant.get_llm_name()
            discussant_config = self.config_list[0][discussant_llm_name]

            messages_r2 = [
                {"role": "system", "content": u1_instruction},
                {"role": "user", "content": f'Opponent ({opponent.name}) statement: "{text1_opponent}"\n\nWhat is your stance on the statement: "{theme}"?'}
            ]

            req_item = {
                "model": discussant_config.get("model", discussant_llm_name),
                "messages": messages_r2,
            }
            if not self.vllm_url and "base_url" in discussant_config:
                req_item["base_url"] = discussant_config["base_url"]
            if "api_key" in discussant_config:
                req_item["api_key"] = discussant_config["api_key"]

            reqs_round2.append(req_item)

        print(f"Generating Round 2 (Discussant answer) for {len(pairs)} pairs...")
        final_text_discussant_list = self.generate_batch(reqs_round2)

        # Round 3: Opponent reaction
        reqs_round3 = []
        for i, (discussant, opponent) in enumerate(pairs):
            _, u2_instruction = instructions[i]
            text1_opponent = text1_opponent_list[i]
            final_text_discussant = final_text_discussant_list[i]

            opponent_llm_name = opponent.get_llm_name()
            opponent_config = self.config_list[0][opponent_llm_name]

            messages_r3 = [
                {"role": "system", "content": u2_instruction},
                {"role": "user", "content": f' What do you think of the following statement?: "{theme}" '},
                {"role": "assistant", "content": text1_opponent},
                {"role": "user", "content": final_text_discussant}
            ]

            req_item = {
                "model": opponent_config.get("model", opponent_llm_name),
                "messages": messages_r3,
            }
            if not self.vllm_url and "base_url" in opponent_config:
                req_item["base_url"] = opponent_config["base_url"]
            if "api_key" in opponent_config:
                req_item["api_key"] = opponent_config["api_key"]

            reqs_round3.append(req_item)

        print(f"Generating Round 3 (Opponent reaction) for {len(pairs)} pairs...")
        text2_opponent_list = self.generate_batch(reqs_round3)

        # Parse opinions transitions
        results = []
        for i, (discussant, opponent) in enumerate(pairs):
            text1_opponent = text1_opponent_list[i]
            final_text_discussant = final_text_discussant_list[i]
            text2_opponent = text2_opponent_list[i]

            op = self.statuses[opponent.name]
            ds = self.statuses[discussant.name]
            new_op_opponent = op
            new_op = ds

            if op == ds:
                results.append((ds, op, text1_opponent, final_text_discussant, text2_opponent))
                continue

            gt = op > ds
            final_words = final_text_discussant.lower().split() if final_text_discussant else []
            opponent_words = text2_opponent.lower().split() if text2_opponent else []

            if "reject" in final_words:
                if gt:
                    new_op = max(ds - 1, self.min_opinion)
                else:
                    new_op = min(ds + 1, self.max_opinion)

                if "accept" in opponent_words:
                    if gt:
                        new_op_opponent = max(op - 1, self.min_opinion)
                    else:
                        new_op_opponent = min(op + 1, self.max_opinion)
                elif "reject" in opponent_words:
                    if gt:
                        new_op_opponent = min(op + 1, self.max_opinion)
                    else:
                        new_op_opponent = max(op - 1, self.min_opinion)

            elif "accept" in final_words:
                if gt:
                    new_op = min(ds + 1, self.max_opinion)
                else:
                    new_op = max(ds - 1, self.min_opinion)
            else:
                new_op = ds

            results.append((new_op, new_op_opponent, text1_opponent, final_text_discussant, text2_opponent))

        return results

    def iteration(self, theme: str) -> object:
        """
        Run an iteration of the simulation in parallel using batched inference.
        """
        pairs = []
        for n1, agent_1 in list(self.agents.agents_iter()):
            if self.meanfield:
                agent_2 = self.agents.get_random_agent()
            else:
                agent_2 = agent_1.get_random_neighbor()
            pairs.append((agent_1, agent_2))

        try:
            results = self.debate_batch(pairs, theme)
        except Exception as e:
            print(f"Error during batched debate: {e}")
            return

        for (agent_1, agent_2), res in zip(pairs, results):
            n1 = agent_1.name
            new_status, new_status_opponent, text1_opponent, final_text_discussant, text2_opponent = res

            if new_status is None:
                new_status = self.statuses[n1]

            if new_status_opponent is None:
                new_status_opponent = self.statuses[agent_2.name]

            original_status = self.statuses[n1]
            self.statuses[n1] = new_status
            agent_1.set_status(new_status)

            original_status_opponent = self.statuses[agent_2.name]
            self.statuses[agent_2.name] = new_status_opponent
            agent_2.set_status(new_status_opponent)

            if self.save_agents_debates:
                yield {
                    "interacting_agents": {
                        "discussant": n1,
                        "discussant_llm": agent_1.get_llm_name(),
                        "opponent": agent_2.name,
                        "opponent_llm": agent_2.get_llm_name(),
                        "discussant_opinion": original_status,
                        "opponent_opinion": original_status_opponent,
                    },
                    "opinion_variation_discussant": new_status - original_status,
                    "opinion_variation_opponent": new_status_opponent - original_status_opponent,
                    "opponent_statement": text1_opponent,
                    "discussant_answer": final_text_discussant,
                    "opponent_answer": text2_opponent,
                    "status": {**self.statuses},
                }
            else:
                yield {
                    "interacting_agents": {
                        "discussant": n1,
                        "discussant_llm": agent_1.get_llm_name(),
                        "opponent": agent_2.name,
                        "opponent_llm": agent_2.get_llm_name(),
                        "discussant_opinion": original_status,
                        "opponent_opinion": original_status_opponent,
                    },
                    "opinion_variation_discussant": new_status - original_status,
                    "opinion_variation_opponent": new_status_opponent - original_status_opponent,
                    "status": {**self.statuses},
                }

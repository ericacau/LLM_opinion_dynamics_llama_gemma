# LLM OD model

This folder contains the simulator, model code, static inputs, prompts, and network files. Install dependencies here:

```bash
cd llm_OD_model
python -m pip install -r requirements.txt
```

Run simulations from this folder with the existing `execution.py` arguments. For example:

```bash
python execution.py meta-llama/Llama-3.2-3B-Instruct 1 theseus_same.json PAH-min0.1-h0.0 100 PAH-min0.1-h0.0.csv --vllm
```

The script reads themes, sample agents, and network files from this folder and writes new run logs to the sibling `../results/` directory. That directory is intentionally empty in this copy.

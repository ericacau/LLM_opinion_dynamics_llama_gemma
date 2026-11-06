# CLI Reference & Execution Configuration

This page provides a comprehensive guide to all configuration parameters, command-line flags, and backend options available in `execution.py`.

---

## Command Syntax

Launch the simulation from the project root directory:

```bash
python execution.py <models> <run_n> <theme_name> <exp_name> <n_agents> [<network>] [options]
```

---

## Detailed Flag & Parameter Reference

### Positional Arguments

#### `models`
*   **Type**: `string` (comma-separated list)
*   **Description**: Specifies the model or models to run (e.g., `llama3.1:latest` or `mistral,gemma`). 
*   **Important**: 
    - The model names specified here must match the `llm_name` parameter defined in the agent profile files (e.g., `"llm_name": "mistral"` inside the agent JSON).
    - If you are running multiple models, separate them with commas without spaces. The simulator will load configurations for each.

#### `run_n`
*   **Type**: `int`
*   **Description**: The number of independent runs of the simulation to execute.
*   **Behavior**: For each run $n \in [0, \text{run\_n}-1]$, it will perform the full simulation and save the output with the suffix `_n`. Useful for statistical validation across multiple stochastic iterations.

#### `theme_name`
*   **Type**: `string`
*   **Description**: The filename of the JSON theme file under the `themes/` folder containing the discussion statement/question (e.g., `theseus_same.json`).

#### `exp_name`
*   **Type**: `string`
*   **Description**: The unique identifier of the experiment configuration (e.g., `PAH-min0.5-h0.0`).
*   **Behavior**: Used by the CLI to determine the directory structure of dataset inputs. It parses the minority size class out of the filename string to determine the directory path (e.g., `PAH-min0.1-h0.25` resolves to the folder `min01/`).

#### `n_agents`
*   **Type**: `int`
*   **Description**: Number of agents to load for the simulation (e.g., `100`).
*   **Behavior**: Used to locate the correct agent JSON profile under `sample_data/`. The target filename is built dynamically: `agents_<exp_name>_<n_agents>_<models>.json`.

#### `network` *(optional)*
*   **Type**: `string`
*   **Description**: Filename of the graph network edge list (e.g., `PAH-min0.5-h0.0.csv`).
*   **Behavior**:
    - If provided, the graph is loaded from `data/min<minority_size>/<network>`. Agent connections are determined by this graph, and neighbors are queried dynamically.
    - If omitted, the simulation defaults to a **meanfield** architecture where agents are paired with any random agent in the pool for debates.

---

### Optional Flags (vLLM & Iteration Tuning)

#### `--vllm`
*   **Type**: Flag (default: `Disabled`)
*   **Description**: **Enables the batched parallel execution engine.**
*   **Details**: 
    - When active, the simulator switches the backend monitor to `MonitorOpinionDistributionVLLM`.
    - Instead of executing agent conversations one by one (sequentially), the batched engine groups all agent interactions and evaluates each round (Opponent statement, Discussant stance, Opponent reaction) concurrently.
    - This flag is critical to increase execution speeds from hours/days to minutes.

#### `--vllm-url`
*   **Type**: `string` (default: `http://localhost:8000/v1`)
*   **Description**: The API endpoint base URL of the OpenAI-compatible model server.
*   **Usage**:
    - If pointing to a running local **vLLM** server, use `http://localhost:8000/v1`.
    - If pointing to local **Ollama**, specify `http://localhost:11434/v1`.
    - When this parameter is set, it overrides the `base_url` parameter defined in the agents' client config lists, ensuring all requests are routed to this endpoint.

#### `--vllm-mode`
*   **Type**: `string` (default: `server`, choices: `['server', 'offline']`)
*   **Description**: The batching implementation backend.
    - `server`: Communicates with an external server using asynchronous HTTP clients. Highly concurrent, does not require a local GPU or the `vllm` library.
    - `offline`: Runs the native Python `vllm` package offline. It loads model weights directly into GPU memory via PyTorch and runs parallel inference using the vLLM engine locally.

#### `-i`, `--iterations`
*   **Type**: `int` (default: `100`)
*   **Description**: The number of simulation iteration steps to execute per run.
*   **Details**: Controls how many iterations are run before writing the final states. A single iteration runs a debate interaction for every agent in the network.

#### `--monitor`
*   **Type**: `string` (default: `MonitorOpinionDistribution`, choices: `['Monitor', 'MonitorOpinionDistribution']`)
*   **Description**: The class structure defining how debates, arguments, and opinion shifts are processed.
    - `MonitorOpinionDistribution`: Includes neighborhood opinion distribution values in agents' templates.
    - `Monitor`: The base Monitor class using simple opinion maps (no neighborhood distribution statistics).
*   **Note**: When `--vllm` is enabled, the CLI automatically maps `Monitor` to its batched equivalent `MonitorVLLM`, and `MonitorOpinionDistribution` to `MonitorOpinionDistributionVLLM`.

---

## Detailed Configuration Matrix

The table below outlines how command-line flags interact to configure the simulation execution backend:

| `--vllm` | `--vllm-mode` | Target Host URL (`--vllm-url`) | Execution Mode | Requirements |
| :--- | :--- | :--- | :--- | :--- |
| **Disabled** | *Ignored* | *Ignored* | **Sequential (Autogen)** | Local Ollama running on port `11434` (Ollama defaults). |
| **Enabled** | `server` | `http://localhost:11434/v1` | **Parallel Async (Ollama)** | Ollama running locally. High concurrency requires setting `OLLAMA_NUM_PARALLEL` environment variable. |
| **Enabled** | `server` | `http://localhost:8000/v1` | **Parallel Async (vLLM Server)** | An external/local vLLM API server running on port `8000`. |
| **Enabled** | `offline` | *Ignored* | **Offline Native (vLLM Engine)** | GPU-capable environment with the `vllm` package installed. HuggingFace credentials to download weights. |

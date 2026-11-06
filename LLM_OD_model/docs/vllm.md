# vLLM Batching & Concurrency

This section details how the parallel batching engine accelerates simulation runtimes, the difference between execution modes, and deployment tips.

---

## Why Batching is Necessary

In sequential simulations, each agent interaction involves up to 4 conversation rounds:
1. **Opponent Statement**: The opponent generates arguments supporting their opinion.
2. **Discussant Stance**: The discussant replies accepting/rejecting the argument.
3. **Opponent Reaction**: The opponent replies to the discussant's acceptance or rejection.

For **100 agents** over **100 iterations**:
- Total debates: `100 * 100 = 10,000`
- Total LLM prompts generated: `10,000 * 3 = 30,000`
- Sequential runtime (at ~2s per generation): **60,000 seconds (~16.6 hours)**.

### Parallel Batching Approach
By enabling `--vllm`, we restructure the simulation loop:
1. At the start of an iteration, all $N$ interacting agent pairs are selected and grouped.
2. **Round 1 (Opponents)**: All $N$ initial statements are sent to the LLM backend concurrently in a single batch.
3. **Round 2 (Discussants)**: Once Round 1 completes, all $N$ answers are sent concurrently.
4. **Round 3 (Opponents)**: Finally, all $N$ opponent reactions are sent concurrently.
5. All opinions are updated and saved together at the end of the iteration step.

This parallel batching architecture is implemented across both supported monitors:
*   `MonitorOpinionDistributionVLLM` (selected when running `--monitor MonitorOpinionDistribution` with `--vllm`)
*   `MonitorVLLM` (selected when running `--monitor Monitor` with `--vllm`)

Since vLLM/Ollama processes concurrent requests in parallel (using continuous batching and page attention), this reduces the total runtime to just **a few minutes** (speedup of over **100x**).

---

## Batching Modes

You can configure the batching engine using the `--vllm-mode` option.

### 1. `server` mode (Default, Recommended)
This mode sends concurrent asynchronous HTTP requests using the `openai.AsyncOpenAI` SDK to a running API server.

*   **Requirements**: An OpenAI-compatible API server running (vLLM, Ollama, or llama.cpp server).
*   **Pros**: 
    - Does not require local vLLM Python packages or GPU packages.
    - Seamlessly supports model name mappings.
    - Automatically formats chat templates on the server side.
*   **Command Example**:
    ```bash
    python execution.py llama3.1:latest 1 theseus_same.json PAH-min0.5-h0.0 100 PAH-min0.5-h0.0.csv --vllm --vllm-url http://localhost:11434/v1 --vllm-mode server
    ```

### 2. `offline` mode
This mode uses the native `vllm.LLM` pipeline to run batched offline inference directly within the Python script execution.

*   **Requirements**: Locally installed GPU-compatible `vllm` library.
*   **Pros**: No server management required. Highly optimized GPU kernel batch execution.
*   **Cons**: Requires downloading the model directly from HuggingFace; takes substantial local GPU memory during runtime.
*   **Command Example**:
    ```bash
    python execution.py meta-llama/Llama-3.1-8B-Instruct 1 theseus_same.json PAH-min0.5-h0.0 100 PAH-min0.5-h0.0.csv --vllm --vllm-mode offline
    ```

---

## Optimizing Ollama for High Concurrency (Server Mode)

By default, **Ollama** runs a single prompt evaluation thread to conserve CPU/GPU resource usage. To support massive batching/concurrency, set the following environment variables before starting Ollama:

=== "macOS"

    ```bash
    # Open terminal and set concurrent requests (e.g. up to 10 concurrent evaluations)
    launchctl setenv OLLAMA_NUM_PARALLEL 10
    
    # Restart Ollama application
    ```

=== "Linux / systemd"

    ```bash
    # Edit the systemd service
    sudo systemctl edit ollama.service
    
    # Add the following lines inside the environment section:
    [Service]
    Environment="OLLAMA_NUM_PARALLEL=10"
    
    # Reload and restart service
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
    ```

# Installation & Setup

Follow these steps to set up the LLM Opinion Dynamics Simulator on your local machine.

---

## 1. Clone the Repository

Navigate to your workspace directory and clone the repository:

```bash
git clone <repository-url>
cd LLM_OD_Llama3.1
```

---

## 2. Environment Setup

It is highly recommended to use a virtual environment (`venv` or `conda`) to manage packages.

=== "Using standard venv"

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

=== "Using Conda"

    ```bash
    conda create -n opinion_dynamics python=3.9 -y
    conda activate opinion_dynamics
    pip install -r requirements.txt
    ```

---

## 3. Package Dependencies

The project relies on the following key Python libraries:

*   **`pyautogen`**: For managing default multi-agent conversational flows.
*   **`networkx`**: For graph topology loading and neighborhood navigation.
*   **`openai`**: The client SDK for communicating with OpenAI-compatible API servers (such as local vLLM or Ollama).
*   **`numpy` & `matplotlib`**: For mathematical processing and trend plotting.
*   **`tqdm`**: For progress bar tracking.

To install additional libraries for high-performance batching:

```bash
pip install httpx aiohttp openai
```

---

## 4. Setting up a Local LLM Server

To run the simulator, you need access to an LLM. You can set up a local model server using **Ollama** or **vLLM**.

### Option A: Ollama (Default sequential backend)

1. Download and install Ollama from [ollama.com](https://ollama.com).
2. Start the Ollama server.
3. Download the required model (e.g., Llama 3.1 or Mistral):

```bash
ollama pull llama3.1:latest
```

### Option B: vLLM Server (High-Performance Batched Backend)

If you have a dedicated GPU, running a vLLM server is highly recommended to take advantage of parallel batching:

```bash
# Install vllm package
pip install vllm

# Spin up the OpenAI-compatible vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000
```

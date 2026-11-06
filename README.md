# LLM opinion dynamics: code, analyses, and data

This repository accompanies the study of LLM opinion dynamics on networks with different population balances and homophily levels. The updated simulator, figures, and validation materials are aligned with **[Zenodo record 22938433](https://zenodo.org/records/22938433)**, DOI **[10.5281/zenodo.22938433](https://doi.org/10.5281/zenodo.22938433)**.

**The raw simulation results are hosted on Zenodo, not in this working tree.** Download [results.zip](https://zenodo.org/records/22938433/files/results.zip?download=1) to inspect the original logs or recompute statistics. You do not need the raw logs or an LLM server to regenerate figures 8–10 from the included statistical outputs.

## Repository contents

| Location | Contents |
| --- | --- |
| `LLM_OD_model/` | Simulator, prompts, example agents, network inputs, and original documentation from `LLM_OD_model.zip`. |
| `analysis/plots/` | Deposited figures, plotting script, and notebook from `analysis.zip`. |
| `analysis/validation/reference_outputs/` | Deposited null-model reference outputs, grouped by parameter setting. |
| `analysis/validation/statistical_outputs/` | Saved statistical results and significant-cell tables. |
| `provenance/` | Zenodo archive checksums and SHA-256 manifest of the imported files. |
| `results/` | Local raw-data directory, populated by extracting the Zenodo download; ignored by Git. |

The historical [`submission` tag](https://github.com/ericacau/LLM_opinion_dynamics_llama_gemma/tree/submission) remains unchanged. This updated layout replaces the older root-level simulator and analysis layout; run the commands below from this repository's root. Historical files remain available in Git history.

## Download the simulation results

The Zenodo record describes **604 JSONL files**, with **302 in `gemma_res_r1/` and 302 in `llama_res_r1/`**. The archive is approximately **4.41 GB compressed**; allow space for roughly 20 GB of extracted materials. Each JSONL line is one JSON record. Filenames encode the experiment, model, run index, and network setting; for example, `_8_PAH-min0.5-h0.75.jsonl` denotes run 8, population balance 0.5, and homophily 0.75.

Download and verify the archive using Python's standard library:

```bash
python scripts/download_results.py --output results.zip
```

The helper checks both the deposited file size and MD5 checksum before completing the download. It does not extract or run anything. Alternatively, download manually using the link above and check:

```text
results.zip
Size: 4408580349 bytes
MD5: e358e70755d7d73e0511848084894a92
```

Extract the archive so both model directories are beneath `results/` (for example, `results/gemma_res_r1/` and `results/llama_res_r1/`). Inspect the archive's top-level directories first to avoid accidentally adding a second `results/` wrapper. The analysis script searches recursively, so an extra wrapper inside `results/` is also supported. Do not commit the archive or extracted logs.

## Regenerate figures from saved statistics

Create an environment and install the analysis dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
python analysis/plots/scripts/opinion_shift_matrices.py --output-dir reproduced_figures
```

The default uses the deposited statistical JSON files and generates:

- Figure 8: `reproduced_figures/figures/probability_matrices_2x2.png`
- Figure 9: `reproduced_figures/figures/facetplot_llama.png`
- Figure 10: `reproduced_figures/figures/facetplot_gemma.png`

This regenerates plots from saved statistics; it does not rerun the simulations or independently recompute their statistical tests. See [reproducibility notes](REPRODUCIBILITY.md) for validation and limitations.

## Recompute statistics from raw logs

After downloading and extracting `results.zip`, run:

```bash
python analysis/plots/scripts/opinion_shift_matrices.py \
  --recompute-statistics \
  --results-dir results \
  --graphs-dir LLM_OD_model/data/graphs \
  --output-dir reproduced_figures
```

The explicit graph path corrects the lowercase `llm_OD_model` default in the deposited script on case-sensitive systems. Defaults select population balance 0.5, the model identifiers `meta-llama_Llama-3.2-3B-Instruct` and `google_gemma-3-4b-it`, 1,000 permutations, significance threshold 0.01, and seed 42. Use `--help` to inspect the available filters. Recomputed statistics depend on the selected logs and settings.

## Run new simulations

The archived simulator and its dependencies are in `LLM_OD_model/`:

```bash
python -m pip install -r LLM_OD_model/requirements.txt
```

New simulations require a configured model backend and matching agent profiles. **The deposited simulation quick start is not ready to run unchanged**: the supplied profiles are under `sample_data/llama/` and identify `llama3.1`, while the example references Llama 3.2; the agent loader also rewrites nested paths. See [the known issues](REPRODUCIBILITY.md#simulation-configuration-limitations) before attempting a new run. The imported scientific code is preserved byte-for-byte rather than silently changing the archived experiment implementation.

The deposited `execution.py` writes new logs to the repository-level `results/` directory, a sibling of `LLM_OD_model/`. The Zenodo description's `LLM_OD_model/results/` path is incorrect for this deposited code.

## Provenance and citation

The code and analysis files were imported from the checksum-verified Zenodo archives. macOS metadata and Python caches were excluded. Check the imported files locally with:

```bash
python scripts/verify_archive_files.py
```

The GitHub README, reproducibility notes, analysis dependency list, download helper, and provenance files are repository additions. They are not presented as files already contained in the deposit. The deposit identifies its license as **CC BY 4.0**; see the [record's license information](https://zenodo.org/records/22938433).

Please cite the archived software/data version using DOI **10.5281/zenodo.22938433**, and cite the associated paper when its bibliographic details are available. Zenodo currently lists older repository names in its title and related links; use the DOI above to identify the exact deposit.

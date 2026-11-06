# Reproducibility and archive alignment

## What this update establishes

The source is Zenodo record **22938433**, DOI **10.5281/zenodo.22938433**. The GitHub `submission` tag and `main` branch both pointed to commit `a7897df92a89bbe3282c4821c4b5ead5ca72fec3` when the alignment was prepared. The tag is preserved; this update reorganizes the current working tree around the two deposited packages.

Both small archives were downloaded and their MD5 checksums verified against the Zenodo API:

| Archive | Bytes | MD5 |
| --- | ---: | --- |
| `LLM_OD_model.zip` | 1105576 | `1c9b75972625148e5dd0abf57a1c130b` |
| `analysis.zip` | 10615142 | `abe3fb61841a2b9cb86723a756a34358` |

All 1,371 imported substantive files match those archives byte-for-byte. The manifest in `provenance/archive-files.json` records their SHA-256 hashes. macOS resource forks, `.DS_Store`, and Python caches are excluded. Repository additions are documented in the README and are outside this import manifest.

Raw results are kept on Zenodo. The file size, checksum, and reported count of 604 JSONL files come from the deposit metadata. The 4.41 GB raw archive was **not downloaded or independently enumerated** during this alignment. Removing raw logs from the current tree does not remove old data from Git history; existing history was not rewritten. A shallow clone limits historical downloads.

## Validation performed

- Verified both downloaded archive checksums and all 1,371 imported file hashes.
- Executed the deposited plotting script using its default saved-statistics mode; all three figure outputs (8, 9, and 10) were generated successfully.
- Checked Python syntax and the README's local links.
- Checked the download helper's checksum validation with a small local fixture; the full results download was not performed.

This validates plotting from saved statistics, not numerical equivalence to a fresh simulation. No LLM inference, full raw-log statistical recomputation, or GPU benchmark was run. Dependency versions used for the plotting check are recorded in `provenance/plotting-validation-environment.json`; the dependency list is not a reconstruction of the original experiment environment.

## Simulation configuration limitations

The simulator source is preserved as deposited. The following issues must be resolved before claiming that the published quick-start simulation command reproduces the archived experiments:

1. `LLM_OD_model/README.md` uses `cd llm_OD_model`, but the actual directory is `LLM_OD_model`.
2. Its example requests `meta-llama/Llama-3.2-3B-Instruct`. The supplied example profiles are in `sample_data/llama/` and declare `llm_name: llama3.1`; matching profiles for the example are not included.
3. `execution.py` constructs agent paths directly under `sample_data/`, whereas the supplied profiles are in its `llama/` subdirectory. Model identifiers containing `/` also affect path construction.
4. `Network.add_agents()` rewrites filenames containing more than two slash-separated components. This conflicts with the absolute paths passed by the deposited `execution.py`.
5. The CLI's default backend URL is `http://localhost:8000/v1`, including without `--vllm`, while parts of the original documentation describe Ollama on port 11434. Set the URL explicitly for the chosen server.
6. The simulator dependencies are only partially pinned, and the archived documentation does not establish an exact historical inference environment. Do not infer reproducibility of model sampling or performance from the saved-figure check.

These are documented rather than silently changing model inputs or scientific behavior. The saved-statistics plotting path works independently of these simulator issues.

## Corrections needed in Zenodo metadata

The published record itself was not modified by this repository update:

- Its description says new simulation logs go to `LLM_OD_model/results/`. The deposited code instead uses the sibling `results/` directory.
- Its title is `ericacau/LLM_OD_homophily: Submission2`, and its related identifiers reference `LLM_OD_Llama3.1`. The current repository is `ericacau/LLM_opinion_dynamics_llama_gemma`.
- After publishing the aligned revision, add its permanent commit/release URL to the deposit's related identifiers and cite the same version in the manuscript.

Updating descriptive metadata can clarify the existing deposit. Changes to the archived code or scientific inputs should be recorded as a new version with an explicit change history.

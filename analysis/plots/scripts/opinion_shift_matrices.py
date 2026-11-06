#!/usr/bin/env python3
"""Reproduce the opinion-shift matrix figures from ex.ipynb.

Defaults reproduce the notebook outputs used for figures 8, 9, and 10:
  - Figure 8: probability_matrices_2x2.png
  - Figure 9: facetplot_llama.png
  - Figure 10: facetplot_gemma.png
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_: object):
        return iterable


PREFIX_ORDER = {
    "": 0,
    "opinion_distr_": 1,
    "reverse_": 2,
    "reverse_opinion_distr_": 3,
}

VARSIGN_LABELS = {
    -1: "Opposing neighbors",
    0: "Mixed",
    1: "Supporting neighbors",
}

DISC_GROUP_ORDER = ["Low (0-3)", "High (4-6)"]
OPP_GROUP_ORDER = ["Low (0-3)", "High (4-6)"]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS = [
    "meta-llama_Llama-3.2-3B-Instruct",
    "google_gemma-3-4b-it",
]
FIGURE8_PREFIXES = ["", "opinion_distr_"]
VARSIGN_PREFIXES = ["opinion_distr_"]
FILENAME_RE = re.compile(r"^(?P<body>.+)_(?P<run_id>\d+)_PAH-min(?P<min>[^-]+)-h(?P<h>.+)\.jsonl$")
MACRO_PAIR_KEYS = [(0, 0), (0, 4), (4, 0), (4, 4)]
MACRO_TRIPLE_KEYS = [(disc, opp, vs) for vs in [-1, 0, 1] for disc, opp in MACRO_PAIR_KEYS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate figures 8, 9, and 10 from the saved opinion-shift JSON files."
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=["8", "9", "10"],
        default=["8", "9", "10"],
        help="Figures to generate. Defaults to all three.",
    )
    parser.add_argument(
        "--probabilities-json",
        default="analysis/validation/statistical_outputs/opinion_shift_matrix_figure8_valid001.json",
        help="Precomputed JSON file used for figure 8 when --use-precomputed-json is set.",
    )
    parser.add_argument(
        "--informed-json",
        default="analysis/validation/statistical_outputs/opinion_shift_matrix_neighborhood_valid001.json",
        help="Saved statistical results used by figures 9 and 10 unless --recompute-statistics is set.",
    )
    parser.add_argument(
        "--use-precomputed-json",
        dest="use_precomputed_json",
        action="store_true",
        default=True,
        help="Plot from saved statistical results (the reproducible default).",
    )
    parser.add_argument(
        "--recompute-statistics",
        dest="use_precomputed_json",
        action="store_false",
        help="Recompute tests from the current simulation logs; values can change if those logs change.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing raw simulation .jsonl files used to redo statistical tests.",
    )
    parser.add_argument(
        "--graphs-dir",
        default="llm_OD_model/data/graphs",
        help="Directory containing graphml files used for neighborhood-condition tests.",
    )
    parser.add_argument(
        "--scenario",
        default="theseus_same",
        help="Scenario name embedded in raw result filenames.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names to include when discovering raw result files.",
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        type=int,
        default=None,
        help="Run IDs to include. Defaults to all matching runs found in --results-dir.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of null-model permutations for the statistical tests.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Significance threshold used to filter matrix cells.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for null-model permutations.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/plots/opinion_shift_matrices",
        help="Directory where output figures are written.",
    )
    parser.add_argument(
        "--min-value",
        type=float,
        default=0.5,
        help="Population balance value to plot. The paper figures use 0.5.",
    )
    parser.add_argument(
        "--figure-8-output",
        default="figures/probability_matrices_2x2.png",
        help="Filename for figure 8.",
    )
    parser.add_argument(
        "--figure-9-output",
        default="figures/facetplot_llama.png",
        help="Filename for figure 9.",
    )
    parser.add_argument(
        "--figure-10-output",
        default="figures/facetplot_gemma.png",
        help="Filename for figure 10.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures. The notebook uses 300 for these FacetGrid figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving.",
    )
    return parser.parse_args()


def resolve_input_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.exists() or path.is_absolute():
        return path

    script_relative = SCRIPT_DIR / path
    if script_relative.exists():
        return script_relative

    return path


def load_json(path: str | Path) -> dict:
    with resolve_input_path(path).open("r") as f:
        return json.load(f)


def parse_tuple_key(value: str) -> tuple[int, ...]:
    return ast.literal_eval(value)


def nanstd(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return np.nan
    return float(np.nanstd(values.to_numpy(dtype=float), ddof=0))


def mean_std_label(mean_value: float, std_value: float) -> str:
    if pd.isna(std_value):
        std_value = 0.0
    return f"{mean_value:.2f}\n±{std_value:.2f}"


def same_float(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) < 1e-12


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.exists() or path.is_absolute():
        return path
    return SCRIPT_DIR / path


def build_filename(
    prefix: str,
    scenario: str,
    model: str,
    run_id: int,
    min_value: float,
    h_value: float,
) -> str:
    return f"{prefix}{scenario}_{model}_{run_id}_PAH-min{min_value}-h{h_value}.jsonl"


def parse_result_filename(filename: str, scenario: str, prefixes: list[str]) -> dict | None:
    match = FILENAME_RE.match(filename)
    if not match:
        return None

    body = match.group("body")
    for prefix in sorted(prefixes, key=len, reverse=True):
        stem_prefixes = [
            f"{prefix}{scenario}_",
            f"theseus_{prefix}{scenario}_",
        ]
        matched_stem = next((stem for stem in stem_prefixes if body.startswith(stem)), None)
        if matched_stem is not None:
            return {
                "prefix": prefix,
                "scenario": scenario,
                "model": body[len(matched_stem) :],
                "run_id": int(match.group("run_id")),
                "min": float(match.group("min")),
                "h": float(match.group("h")),
            }
    return None


def discover_result_specs(
    results_dir: str | Path,
    prefixes: list[str],
    scenario: str,
    models: list[str],
    min_value: float,
    run_ids: list[int] | None,
) -> list[dict]:
    results_dir = resolve_repo_path(results_dir)
    specs = []
    for path in results_dir.rglob("*.jsonl"):
        params = parse_result_filename(path.name, scenario, prefixes)
        if params is None:
            continue
        if params["model"] not in models:
            continue
        if not same_float(params["min"], min_value):
            continue
        if run_ids is not None and params["run_id"] not in run_ids:
            continue
        specs.append({"filename": path.name, "path": path, "params": params})

    return sorted(
        specs,
        key=lambda spec: (
            spec["params"]["model"],
            PREFIX_ORDER.get(spec["params"]["prefix"], 99),
            spec["params"]["min"],
            spec["params"]["h"],
            spec["params"]["run_id"],
        ),
    )


def model_display_name(model: str) -> str:
    model_lower = model.lower()
    if "llama" in model_lower:
        return "llama"
    if "gemma" in model_lower:
        return "gemma"
    return model


def moved_toward_opponent(disc_opinion: int, opp_opinion: int, opinion_var: int) -> int:
    if opp_opinion > disc_opinion and opinion_var > 0:
        return 1
    if opp_opinion < disc_opinion and opinion_var < 0:
        return 1
    return 0


def opinion_group_value(opinion: int) -> int:
    return 0 if opinion <= 3 else 4


def observed_macro_mean(values_by_run: list[dict], key: tuple[int, ...]) -> float:
    values = [run_values.get(key, np.nan) for run_values in values_by_run]
    values = np.array(values, dtype=float)
    if np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))


def null_macro_means(nulls_by_run: list[dict], key: tuple[int, ...], n_permutations: int) -> np.ndarray:
    null_values = []
    for permutation_idx in range(n_permutations):
        values = [
            run_nulls.get(key, [np.nan] * n_permutations)[permutation_idx]
            for run_nulls in nulls_by_run
        ]
        values = np.array(values, dtype=float)
        null_values.append(np.nanmean(values) if not np.all(np.isnan(values)) else np.nan)
    return np.array(null_values, dtype=float)


def two_tailed_permutation_result(observed: float, null_values: np.ndarray, alpha: float) -> dict | None:
    null_values = null_values[~np.isnan(null_values)]
    if pd.isna(observed) or len(null_values) == 0:
        return None

    null_mean = float(np.mean(null_values))
    null_std = float(np.std(null_values))
    if observed >= null_mean:
        p_value = float(np.mean(null_values >= observed))
    else:
        p_value = float(np.mean(null_values <= observed))
    p_value = min(2 * p_value, 1.0)

    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "p_value": p_value,
        "significant": p_value < alpha,
        "effect_size": (observed - null_mean) / null_std if null_std > 0 else 0,
    }


def compute_pvalues(
    observed_probs: dict,
    null_distribution: dict,
    alpha: float = 0.01,
) -> dict:
    results = {}
    for key, obs_prob in observed_probs.items():
        if key not in null_distribution:
            continue

        null_probs = np.array(null_distribution[key])
        null_mean = np.mean(null_probs)
        null_std = np.std(null_probs)

        if obs_prob >= null_mean:
            p_value = np.mean(null_probs >= obs_prob)
        else:
            p_value = np.mean(null_probs <= obs_prob)

        p_value = min(2 * p_value, 1.0)
        results[key] = {
            "observed": obs_prob,
            "null_mean": null_mean,
            "null_std": null_std,
            "p_value": p_value,
            "significant": p_value < alpha,
            "effect_size": (obs_prob - null_mean) / null_std if null_std > 0 else 0,
        }
    return results


def load_interactions(filepath: str | Path) -> list[dict]:
    interactions = []
    with Path(filepath).open("r") as f:
        for line in f:
            record = json.loads(line)
            if "interacting_agents" not in record:
                continue
            agents = record["interacting_agents"]
            interactions.append(
                {
                    "disc_opinion": agents["discussant_opinion"],
                    "opp_opinion": agents["opponent_opinion"],
                    "opinion_var": record["opinion_variation_discussant"],
                }
            )
    return interactions


def compute_probabilities_from_interactions(interactions: list[dict]) -> dict:
    counts = defaultdict(lambda: {"toward": 0, "total": 0})
    for inter in interactions:
        key = (inter["disc_opinion"], inter["opp_opinion"])
        counts[key]["total"] += 1
        counts[key]["toward"] += moved_toward_opponent(
            inter["disc_opinion"],
            inter["opp_opinion"],
            inter["opinion_var"],
        )

    return {
        key: data["toward"] / data["total"] if data["total"] > 0 else 0.0
        for key, data in counts.items()
    }


def compute_macro_probabilities_from_interactions(interactions: list[dict]) -> dict:
    counts = defaultdict(lambda: {"toward": 0, "total": 0})
    for inter in interactions:
        key = (
            opinion_group_value(inter["disc_opinion"]),
            opinion_group_value(inter["opp_opinion"]),
        )
        counts[key]["total"] += 1
        counts[key]["toward"] += moved_toward_opponent(
            inter["disc_opinion"],
            inter["opp_opinion"],
            inter["opinion_var"],
        )

    return {
        key: data["toward"] / data["total"] if data["total"] > 0 else np.nan
        for key, data in counts.items()
    }


def generate_null_distribution(
    interactions: list[dict],
    n_permutations: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    opinion_vars = [inter["opinion_var"] for inter in interactions]
    null_probs = defaultdict(list)

    for _ in range(n_permutations):
        shuffled_vars = rng.permutation(opinion_vars)
        counts = defaultdict(lambda: {"toward": 0, "total": 0})
        for inter, opinion_var in zip(interactions, shuffled_vars):
            key = (inter["disc_opinion"], inter["opp_opinion"])
            counts[key]["total"] += 1
            counts[key]["toward"] += moved_toward_opponent(
                inter["disc_opinion"],
                inter["opp_opinion"],
                opinion_var,
            )

        for key, data in counts.items():
            null_probs[key].append(data["toward"] / data["total"] if data["total"] > 0 else 0.0)

    return dict(null_probs)


def generate_macro_null_distribution(
    interactions: list[dict],
    n_permutations: int,
    seed: int,
    progress_label: str | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    opinion_vars = np.asarray([inter["opinion_var"] for inter in interactions])
    disc = np.asarray([inter["disc_opinion"] for inter in interactions])
    opp = np.asarray([inter["opp_opinion"] for inter in interactions])
    group_codes = (disc > 3).astype(np.int8) * 2 + (opp > 3).astype(np.int8)
    toward_up = opp > disc
    toward_down = opp < disc
    grouped_indices = {
        key: np.flatnonzero(group_codes == index)
        for index, key in enumerate(MACRO_PAIR_KEYS)
    }
    null_probs = {key: [] for key in MACRO_PAIR_KEYS}

    for _ in tqdm(
        range(n_permutations),
        desc=progress_label,
        leave=False,
        unit="perm",
    ):
        shuffled_vars = rng.permutation(opinion_vars)
        toward = (toward_up & (shuffled_vars > 0)) | (toward_down & (shuffled_vars < 0))
        for key in MACRO_PAIR_KEYS:
            indices = grouped_indices[key]
            null_probs[key].append(
                float(np.mean(toward[indices])) if len(indices) else np.nan
            )

    return null_probs


def run_pairwise_null_model(
    filepath: str | Path,
    n_permutations: int,
    seed: int,
) -> dict | None:
    interactions = load_interactions(filepath)
    if not interactions:
        return None

    observed_probs = compute_probabilities_from_interactions(interactions)
    null_dist = generate_null_distribution(interactions, n_permutations, seed)
    return {
        "n_interactions": len(interactions),
        "observed_probs": observed_probs,
        "statistical_tests": compute_pvalues(observed_probs, null_dist),
    }


def filter_significant_results(null_model_results: dict, alpha: float) -> dict:
    filtered = {}
    for filename, res in null_model_results.items():
        significant_pairs = {
            str(pair): {
                "observed_prob": stats["observed"],
                "null_mean": stats["null_mean"],
                "p_value": stats["p_value"],
                "effect_size": stats["effect_size"],
            }
            for pair, stats in res["statistical_tests"].items()
            if stats["p_value"] < alpha
        }
        if significant_pairs:
            filtered[filename] = {
                "params": res["params"],
                "n_interactions": res["n_interactions"],
                "significant_pairs": significant_pairs,
                "n_significant": len(significant_pairs),
                "n_total_pairs": len(res["statistical_tests"]),
            }
    return filtered


def recompute_pairwise_filtered_results(
    results_dir: str | Path,
    scenario: str,
    models: list[str],
    min_value: float,
    run_ids: list[int] | None,
    n_permutations: int,
    seed: int,
    alpha: float,
) -> dict:
    specs = discover_result_specs(
        results_dir,
        FIGURE8_PREFIXES,
        scenario,
        models,
        min_value,
        run_ids,
    )
    grouped_specs = defaultdict(list)
    for spec in specs:
        params = spec["params"]
        grouped_specs[
            (
                params["model"],
                params["prefix"],
                params["scenario"],
                params["min"],
                params["h"],
            )
        ].append(spec)

    filtered = {}
    grouped_values = list(grouped_specs.values())
    for group_idx, group_specs in enumerate(
        tqdm(grouped_values, desc="Figure 8 run-level tests", unit="matrix")
    ):
        valid_specs = []
        n_interactions_by_run = []
        observed_by_run = []
        nulls_by_run = []
        for run_idx, spec in enumerate(group_specs):
            interactions = load_interactions(spec["path"])
            if not interactions:
                continue
            valid_specs.append(spec)
            n_interactions_by_run.append(len(interactions))
            observed_by_run.append(compute_macro_probabilities_from_interactions(interactions))
            nulls_by_run.append(
                generate_macro_null_distribution(
                    interactions,
                    n_permutations,
                    seed + 1009 * group_idx + run_idx,
                    progress_label=(
                        "Fig 8 "
                        f"{spec['params']['model']} "
                        f"{spec['params']['prefix'] or 'base'} "
                        f"h={spec['params']['h']} "
                        f"run={spec['params']['run_id']}"
                    ),
                )
            )

        if not observed_by_run:
            continue

        significant_pairs = {}
        for key in MACRO_PAIR_KEYS:
            observed = observed_macro_mean(observed_by_run, key)
            null_values = null_macro_means(nulls_by_run, key, n_permutations)
            stats = two_tailed_permutation_result(observed, null_values, alpha)
            if stats is not None and stats["significant"]:
                significant_pairs[str(key)] = {
                    "observed_prob": stats["observed"],
                    "null_mean": stats["null_mean"],
                    "p_value": stats["p_value"],
                    "effect_size": stats["effect_size"],
                }

        if not significant_pairs:
            continue

        for run_idx, spec in enumerate(valid_specs):
            run_pairs = {}
            for key_str in significant_pairs:
                key = parse_tuple_key(key_str)
                value = observed_by_run[run_idx].get(key, np.nan)
                if not pd.isna(value):
                    run_pairs[key_str] = {
                        **significant_pairs[key_str],
                        "observed_prob": value,
                    }

            if run_pairs:
                filtered[spec["filename"]] = {
                    "params": spec["params"],
                    "n_interactions": n_interactions_by_run[run_idx],
                    "significant_pairs": run_pairs,
                    "n_significant": len(run_pairs),
                    "n_total_pairs": len(MACRO_PAIR_KEYS),
                }

    return filtered


def load_graph(graphs_dir: str | Path, min_value: float, h_value: float):
    import networkx as nx

    graph_path = resolve_repo_path(graphs_dir) / f"PAH-min{min_value}-h{h_value}.graphml"
    return nx.read_graphml(graph_path)


def get_majority_neighbor_opinion(graph, agent_id: str, current_status: dict) -> tuple[int | None, bool]:
    node_id = agent_id[1:]
    if node_id not in graph:
        return None, False

    neighbor_opinions = []
    for neighbor in graph.neighbors(node_id):
        neighbor_agent = f"a{neighbor}"
        if neighbor_agent in current_status:
            neighbor_opinions.append(current_status[neighbor_agent])

    if not neighbor_opinions:
        return None, False

    most_common = Counter(neighbor_opinions).most_common()
    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
        return most_common[0][0], True
    return None, False


def compute_varsign(
    disc_opinion_before: int,
    opinion_variation: int,
    majority_opinion: int | None,
    has_majority: bool,
) -> int:
    if not has_majority or opinion_variation == 0:
        return 0

    disc_opinion_after = disc_opinion_before + opinion_variation
    dist_before = abs(disc_opinion_before - majority_opinion)
    dist_after = abs(disc_opinion_after - majority_opinion)

    if dist_after < dist_before:
        return 1
    if dist_after > dist_before:
        return -1
    return 0


def load_interactions_with_varsign(filepath: str | Path, graph) -> list[dict]:
    interactions = []
    last_status = None
    iteration_start_status = None
    active_iteration = None

    with Path(filepath).open("r") as f:
        for line in f:
            record = json.loads(line)
            current_status = record.get("status", {})

            if "interacting_agents" not in record:
                last_status = current_status.copy()
                continue

            iteration = record.get("iteration")
            if iteration != active_iteration:
                iteration_start_status = (
                    last_status.copy() if last_status is not None else current_status.copy()
                )
                active_iteration = iteration

            agents = record["interacting_agents"]
            discussant = agents["discussant"]
            disc_opinion = agents["discussant_opinion"]
            opp_opinion = agents["opponent_opinion"]
            opinion_var = record["opinion_variation_discussant"]

            if iteration_start_status is not None:
                majority_op, has_majority = get_majority_neighbor_opinion(
                    graph,
                    discussant,
                    iteration_start_status,
                )
                varsign = compute_varsign(disc_opinion, opinion_var, majority_op, has_majority)
            else:
                varsign = 0

            interactions.append(
                {
                    "disc_opinion": disc_opinion,
                    "opp_opinion": opp_opinion,
                    "varsign": varsign,
                    "toward": moved_toward_opponent(disc_opinion, opp_opinion, opinion_var),
                }
            )
            # Each event in one iteration was prompted from the same pre-update
            # state. The logged status is committed incrementally, so retain it
            # only as the next iteration's starting state.
            last_status = current_status.copy()

    return interactions


def compute_probs_from_interactions_with_varsign(interactions: list[dict]) -> dict:
    counts = defaultdict(lambda: {"toward": 0, "total": 0})
    for inter in interactions:
        key = (inter["disc_opinion"], inter["opp_opinion"], inter["varsign"])
        counts[key]["total"] += 1
        counts[key]["toward"] += inter["toward"]

    return {
        key: data["toward"] / data["total"] if data["total"] > 0 else 0.0
        for key, data in counts.items()
    }


def compute_macro_probs_from_interactions_with_varsign(interactions: list[dict]) -> dict:
    counts = defaultdict(lambda: {"toward": 0, "total": 0})
    for inter in interactions:
        key = (
            opinion_group_value(inter["disc_opinion"]),
            opinion_group_value(inter["opp_opinion"]),
            inter["varsign"],
        )
        counts[key]["total"] += 1
        counts[key]["toward"] += inter["toward"]

    return {
        key: data["toward"] / data["total"] if data["total"] > 0 else np.nan
        for key, data in counts.items()
    }


def generate_null_distribution_varsign(
    interactions: list[dict],
    n_permutations: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    varsigns = [inter["varsign"] for inter in interactions]
    null_probs = defaultdict(list)

    for _ in range(n_permutations):
        shuffled_varsigns = rng.permutation(varsigns)
        counts = defaultdict(lambda: {"toward": 0, "total": 0})
        for inter, varsign in zip(interactions, shuffled_varsigns):
            key = (inter["disc_opinion"], inter["opp_opinion"], varsign)
            counts[key]["total"] += 1
            counts[key]["toward"] += inter["toward"]

        for key, data in counts.items():
            null_probs[key].append(data["toward"] / data["total"] if data["total"] > 0 else 0.0)

    return dict(null_probs)


def generate_macro_null_distribution_varsign(
    interactions: list[dict],
    n_permutations: int,
    seed: int,
    progress_label: str | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    varsigns = np.asarray([inter["varsign"] for inter in interactions], dtype=np.int8)
    disc = np.asarray([inter["disc_opinion"] for inter in interactions])
    opp = np.asarray([inter["opp_opinion"] for inter in interactions])
    group_codes = (disc > 3).astype(np.int8) * 2 + (opp > 3).astype(np.int8)
    toward = np.asarray([inter["toward"] for inter in interactions], dtype=np.int8)
    grouped_indices = {
        key: np.flatnonzero(group_codes == index)
        for index, key in enumerate(MACRO_PAIR_KEYS)
    }
    null_probs = {key: [] for key in MACRO_TRIPLE_KEYS}

    for _ in tqdm(
        range(n_permutations),
        desc=progress_label,
        leave=False,
        unit="perm",
    ):
        shuffled_varsigns = rng.permutation(varsigns)
        for pair_index, pair_key in enumerate(MACRO_PAIR_KEYS):
            indices = grouped_indices[pair_key]
            if not len(indices):
                for varsign in (-1, 0, 1):
                    null_probs[(*pair_key, varsign)].append(np.nan)
                continue
            shuffled = shuffled_varsigns[indices]
            toward_group = toward[indices]
            for varsign in (-1, 0, 1):
                selected = shuffled == varsign
                probability = (
                    float(np.mean(toward_group[selected]))
                    if np.any(selected)
                    else np.nan
                )
                null_probs[(*pair_key, varsign)].append(probability)

    return null_probs


def run_varsign_null_model(
    filepath: str | Path,
    graph,
    n_permutations: int,
    seed: int,
) -> dict | None:
    interactions = load_interactions_with_varsign(filepath, graph)
    if not interactions:
        return None

    observed_probs = compute_probs_from_interactions_with_varsign(interactions)
    null_dist = generate_null_distribution_varsign(interactions, n_permutations, seed)
    return {
        "n_interactions": len(interactions),
        "observed_probs": observed_probs,
        "statistical_tests": compute_pvalues(observed_probs, null_dist),
    }


def filter_significant_varsign(null_model_results: dict, alpha: float) -> dict:
    filtered = {}
    for filename, res in null_model_results.items():
        significant_triples = {
            str(key): {
                "observed_prob": stats["observed"],
                "null_mean": stats["null_mean"],
                "p_value": stats["p_value"],
                "effect_size": stats["effect_size"],
            }
            for key, stats in res["statistical_tests"].items()
            if stats["p_value"] < alpha
        }
        if significant_triples:
            filtered[filename] = {
                "params": res["params"],
                "n_interactions": res["n_interactions"],
                "significant_triples": significant_triples,
                "n_significant": len(significant_triples),
                "n_total": len(res["statistical_tests"]),
            }
    return filtered


def recompute_varsign_filtered_results(
    results_dir: str | Path,
    graphs_dir: str | Path,
    scenario: str,
    models: list[str],
    min_value: float,
    run_ids: list[int] | None,
    n_permutations: int,
    seed: int,
    alpha: float,
) -> dict:
    specs = discover_result_specs(
        results_dir,
        VARSIGN_PREFIXES,
        scenario,
        models,
        min_value,
        run_ids,
    )
    graph_cache = {}
    grouped_specs = defaultdict(list)
    for spec in specs:
        params = spec["params"]
        grouped_specs[
            (
                params["model"],
                params["prefix"],
                params["scenario"],
                params["min"],
                params["h"],
            )
        ].append(spec)

    filtered = {}
    grouped_values = list(grouped_specs.values())
    for group_idx, group_specs in enumerate(
        tqdm(grouped_values, desc="Figures 9/10 run-level tests", unit="matrix")
    ):
        valid_specs = []
        observed_by_run = []
        nulls_by_run = []
        interactions_by_run = []
        for run_idx, spec in enumerate(group_specs):
            graph_key = (spec["params"]["min"], spec["params"]["h"])
            if graph_key not in graph_cache:
                graph_cache[graph_key] = load_graph(graphs_dir, *graph_key)

            interactions = load_interactions_with_varsign(spec["path"], graph_cache[graph_key])
            if not interactions:
                continue

            valid_specs.append(spec)
            interactions_by_run.append(interactions)
            observed_by_run.append(compute_macro_probs_from_interactions_with_varsign(interactions))
            nulls_by_run.append(
                generate_macro_null_distribution_varsign(
                    interactions,
                    n_permutations,
                    seed + 1009 * group_idx + run_idx,
                    progress_label=(
                        "Figs 9/10 "
                        f"{spec['params']['model']} "
                        f"h={spec['params']['h']} "
                        f"run={spec['params']['run_id']}"
                    ),
                )
            )

        if not observed_by_run:
            continue

        significant_triples = {}
        for key in MACRO_TRIPLE_KEYS:
            observed = observed_macro_mean(observed_by_run, key)
            null_values = null_macro_means(nulls_by_run, key, n_permutations)
            stats = two_tailed_permutation_result(observed, null_values, alpha)
            if stats is not None and stats["significant"]:
                significant_triples[str(key)] = {
                    "observed_prob": stats["observed"],
                    "null_mean": stats["null_mean"],
                    "p_value": stats["p_value"],
                    "effect_size": stats["effect_size"],
                }

        if not significant_triples:
            continue

        for run_idx, spec in enumerate(valid_specs):
            run_triples = {}
            for key_str in significant_triples:
                key = parse_tuple_key(key_str)
                value = observed_by_run[run_idx].get(key, np.nan)
                if not pd.isna(value):
                    run_triples[key_str] = {
                        **significant_triples[key_str],
                        "observed_prob": value,
                    }

            if run_triples:
                filtered[spec["filename"]] = {
                    "params": spec["params"],
                    "n_interactions": len(interactions_by_run[run_idx]),
                    "significant_triples": run_triples,
                    "n_significant": len(run_triples),
                    "n_total": len(MACRO_TRIPLE_KEYS),
                }

    return filtered


def sorted_filenames(data: dict, min_value: float | None = None) -> list[str]:
    filenames = []
    for filename, file_data in data.items():
        params = file_data["params"]
        if min_value is not None and not same_float(params["min"], min_value):
            continue
        filenames.append(filename)

    return sorted(
        filenames,
        key=lambda filename: (
            data[filename]["params"]["model"],
            PREFIX_ORDER.get(data[filename]["params"]["prefix"], 99),
            data[filename]["params"]["min"],
            data[filename]["params"]["h"],
        ),
    )


def output_path(output_dir: str | Path, filename: str) -> Path:
    output_dir = Path(output_dir).expanduser()
    path = output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: dict) -> None:
    with path.open("w") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)


def build_figure8_dataframe(filtered_data: dict, min_value: float) -> pd.DataFrame:
    rows = []

    for filename in sorted_filenames(filtered_data, min_value=min_value):
        file_data = filtered_data[filename]
        params = file_data["params"]
        quadrants = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}

        for key_str, d in file_data["significant_pairs"].items():
            disc_op, opp_op = parse_tuple_key(key_str)
            disc_group = 0 if disc_op <= 3 else 1
            opp_group = 0 if opp_op <= 3 else 1
            quadrants[(disc_group, opp_group)].append(d["observed_prob"])

        prefix_label = params["prefix"] if params["prefix"] else "base"
        model_label = model_display_name(params["model"])

        for (disc_group, opp_group), vals in quadrants.items():
            prob = np.median(vals) if vals else np.nan
            rows.append(
                {
                    "model": params["model"],
                    "model_label": model_label,
                    "prefix": prefix_label,
                    "row_label": f"{model_label}\n{prefix_label}",
                    "col_label": f"min={params['min']}, h={params['h']}",
                    "min": params["min"],
                    "h": params["h"],
                    "disc_group": disc_group,
                    "opp_group": opp_group,
                    "prob": prob,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No figure 8 records found for min={min_value}.")

    group_cols = [
        "model",
        "model_label",
        "prefix",
        "row_label",
        "col_label",
        "min",
        "h",
        "disc_group",
        "opp_group",
    ]
    df = (
        df.groupby(group_cols, dropna=False, observed=True)["prob"]
        .agg(prob="mean", prob_std=nanstd, n_realizations="count")
        .reset_index()
    )
    row_order = sorted(
        df["row_label"].dropna().unique(),
        key=lambda label: (
            0 if label.startswith("llama") else 1 if label.startswith("gemma") else 2,
            0 if label.endswith("base") else 1,
            label,
        ),
    )
    df["row_label"] = pd.Categorical(df["row_label"], categories=row_order, ordered=True)
    df["col_label"] = pd.Categorical(df["col_label"], ordered=True)
    return df


def plot_figure8(
    filtered_data: dict,
    min_value: float,
    output_file: Path,
    dpi: int,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    df = build_figure8_dataframe(filtered_data, min_value)

    def facet_heatmap(data: pd.DataFrame, **_: object) -> None:
        ax = plt.gca()
        mat = (
            data.pivot(index="disc_group", columns="opp_group", values="prob")
            .reindex(index=[0, 1], columns=[0, 1])
        )
        std_mat = (
            data.pivot(index="disc_group", columns="opp_group", values="prob_std")
            .reindex(index=[0, 1], columns=[0, 1])
        )

        sns.heatmap(
            mat,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar=False,
            square=True,
            linewidths=0.5,
            linecolor="white",
            annot=False,
        )

        for i in [0, 1]:
            for j in [0, 1]:
                val = mat.loc[i, j]
                if pd.notna(val):
                    std_val = std_mat.loc[i, j]
                    color = "white" if val > 0.6 else "black"
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        mean_std_label(val, std_val),
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color=color,
                    )

        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(["Low\n(0-3)", "High\n(4-6)"], fontsize=8)
        ax.set_yticklabels(["Low\n(0-3)", "High\n(4-6)"], fontsize=8, rotation=0)
        ax.set_xlabel("Opponent Opinion", fontsize=9)
        ax.set_ylabel("Discussant Opinion", fontsize=9)

    sns.set_theme(style="white")
    grid = sns.FacetGrid(
        df,
        row="row_label",
        col="col_label",
        height=2.5,
        aspect=1,
        sharex=True,
        sharey=True,
        margin_titles=True,
        despine=False,
    )
    grid.map_dataframe(facet_heatmap)
    grid.set_titles(col_template="{col_name}", row_template="{row_name}")

    for ax in grid.axes[:, -1]:
        ax.yaxis.set_label_position

    plt.tight_layout()
    grid.savefig(output_file, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(grid.fig)


def create_2x2_matrix_by_varsign(
    significant_triples: dict,
    target_varsign: int,
) -> np.ndarray:
    quadrants = {
        (0, 0): [],
        (0, 1): [],
        (1, 0): [],
        (1, 1): [],
    }

    for key_str, data in significant_triples.items():
        disc_op, opp_op, varsign = parse_tuple_key(key_str)
        if varsign != target_varsign:
            continue
        disc_group = 0 if disc_op <= 3 else 1
        opp_group = 0 if opp_op <= 3 else 1
        quadrants[(disc_group, opp_group)].append(data["observed_prob"])

    matrix = np.full((2, 2), np.nan)
    for (i, j), values in quadrants.items():
        if values:
            matrix[i, j] = np.mean(values)
    return matrix


def build_varsign_dataframe(filtered_varsign: dict, min_value: float) -> pd.DataFrame:
    records = []

    for filename in sorted_filenames(filtered_varsign, min_value=min_value):
        file_data = filtered_varsign[filename]
        params = file_data["params"]

        for vs in [-1, 0, 1]:
            matrix = create_2x2_matrix_by_varsign(file_data["significant_triples"], vs)
            for i, disc_label in enumerate(DISC_GROUP_ORDER):
                for j, opp_label in enumerate(OPP_GROUP_ORDER):
                    records.append(
                        {
                            "model": params["model"],
                            "model_label": model_display_name(params["model"]),
                            "prefix": params["prefix"] if params["prefix"] else "base",
                            "min": params["min"],
                            "h": params["h"],
                            "varsign": vs,
                            "varsign_label": VARSIGN_LABELS[vs],
                            "disc_group": disc_label,
                            "opp_group": opp_label,
                            "prob": matrix[i, j],
                        }
                    )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No varsign records found for min={min_value}.")

    df["row_label"] = df.apply(lambda row: f"h={row['h']}", axis=1)
    group_cols = [
        "model",
        "model_label",
        "prefix",
        "min",
        "h",
        "varsign",
        "varsign_label",
        "disc_group",
        "opp_group",
        "row_label",
    ]
    df = (
        df.groupby(group_cols, dropna=False, observed=True)["prob"]
        .agg(prob="mean", prob_std=nanstd, n_realizations="count")
        .reset_index()
    )
    return df


def plot_model_varsign_grid(
    df_heat: pd.DataFrame,
    model_contains: str,
    output_file: Path,
    dpi: int,
    show: bool,
    adjust_top: bool,
) -> None:
    import matplotlib.pyplot as plt

    model_df = df_heat[df_heat["model"].str.contains(model_contains, case=False, regex=False)]
    if model_df.empty:
        raise ValueError(f"No records found for model containing {model_contains!r}.")

    grid = sns.FacetGrid(
        model_df,
        row="varsign_label",
        col="row_label",
        margin_titles=True,
        despine=False,
    )

    def heatmap(data: pd.DataFrame, color: object, **kws: object) -> None:
        data_pivot = data.pivot(
            index="disc_group",
            columns="opp_group",
            values="prob",
        )
        data_pivot = data_pivot.reindex(columns=OPP_GROUP_ORDER, index=DISC_GROUP_ORDER)
        std_pivot = data.pivot(
            index="disc_group",
            columns="opp_group",
            values="prob_std",
        )
        std_pivot = std_pivot.reindex(columns=OPP_GROUP_ORDER, index=DISC_GROUP_ORDER)
        sns.heatmap(
            data_pivot,
            annot=False,
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar=False,
            linewidths=0.5,
            linecolor="gray",
            **kws,
        )
        ax = plt.gca()
        for i, disc_group in enumerate(DISC_GROUP_ORDER):
            for j, opp_group in enumerate(OPP_GROUP_ORDER):
                val = data_pivot.loc[disc_group, opp_group]
                if pd.notna(val):
                    std_val = std_pivot.loc[disc_group, opp_group]
                    color = "white" if val > 0.6 else "black"
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        mean_std_label(val, std_val),
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color=color,
                    )
        plt.xlabel("Opponent")
        plt.ylabel("Discussant")

    grid.map_dataframe(heatmap)
    grid.set_titles(row_template="{row_name}", col_template="{col_name}")
    if adjust_top:
        grid.fig.subplots_adjust(top=0.9)

    grid.savefig(output_file, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(grid.fig)


def configure_matplotlib(show: bool) -> None:
    if not show:
        import matplotlib

        matplotlib.use("Agg")


def generate_figures(
    *,
    figures: list[str] | tuple[str, ...] = ("8", "9", "10"),
    probabilities_json: str | Path = "analysis/validation/statistical_outputs/opinion_shift_matrix_figure8_valid001.json",
    informed_json: str | Path = "analysis/validation/statistical_outputs/opinion_shift_matrix_neighborhood_valid001.json",
    use_precomputed_json: bool = False,
    results_dir: str | Path = "results",
    graphs_dir: str | Path = "llm_OD_model/data/graphs",
    scenario: str = "theseus_same",
    models: list[str] | tuple[str, ...] = tuple(DEFAULT_MODELS),
    run_ids: list[int] | tuple[int, ...] | None = None,
    n_permutations: int = 1000,
    alpha: float = 0.01,
    seed: int = 42,
    output_dir: str | Path = "analysis/plots/opinion_shift_matrices",
    min_value: float = 0.5,
    figure_8_output: str = "figures/probability_matrices_2x2.png",
    figure_9_output: str = "figures/facetplot_llama.png",
    figure_10_output: str = "figures/facetplot_gemma.png",
    dpi: int = 300,
    show: bool = False,
) -> list[Path]:
    """Generate the requested figures and return their output paths."""
    configure_matplotlib(show)

    generated: list[Path] = []

    if "8" in figures:
        if use_precomputed_json:
            probabilities = load_json(probabilities_json)
        else:
            probabilities = recompute_pairwise_filtered_results(
                results_dir=results_dir,
                scenario=scenario,
                models=list(models),
                min_value=min_value,
                run_ids=list(run_ids) if run_ids is not None else None,
                n_permutations=n_permutations,
                seed=seed,
                alpha=alpha,
            )
        save_json(
            output_path(output_dir, "opinion_shift_matrix_figure8_valid001.json"),
            probabilities,
        )
        path = output_path(output_dir, figure_8_output)
        plot_figure8(probabilities, min_value, path, dpi, show)
        generated.append(path)

    if "9" in figures or "10" in figures:
        if use_precomputed_json:
            informed = load_json(informed_json)
        else:
            informed = recompute_varsign_filtered_results(
                results_dir=results_dir,
                graphs_dir=graphs_dir,
                scenario=scenario,
                models=list(models),
                min_value=min_value,
                run_ids=list(run_ids) if run_ids is not None else None,
                n_permutations=n_permutations,
                seed=seed,
                alpha=alpha,
            )
        save_json(
            output_path(output_dir, "opinion_shift_matrix_neighborhood_valid001.json"),
            informed,
        )
        df_heat = build_varsign_dataframe(informed, min_value)

        if "9" in figures:
            path = output_path(output_dir, figure_9_output)
            plot_model_varsign_grid(df_heat, "llama", path, dpi, show, adjust_top=True)
            generated.append(path)

        if "10" in figures:
            path = output_path(output_dir, figure_10_output)
            plot_model_varsign_grid(df_heat, "gemma", path, dpi, show, adjust_top=False)
            generated.append(path)

    return generated


def main() -> None:
    args = parse_args()
    generated = generate_figures(
        figures=args.figures,
        probabilities_json=args.probabilities_json,
        informed_json=args.informed_json,
        use_precomputed_json=args.use_precomputed_json,
        results_dir=args.results_dir,
        graphs_dir=args.graphs_dir,
        scenario=args.scenario,
        models=args.models,
        run_ids=args.run_ids,
        n_permutations=args.n_permutations,
        alpha=args.alpha,
        seed=args.seed,
        output_dir=args.output_dir,
        min_value=args.min_value,
        figure_8_output=args.figure_8_output,
        figure_9_output=args.figure_9_output,
        figure_10_output=args.figure_10_output,
        dpi=args.dpi,
        show=args.show,
    )

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()

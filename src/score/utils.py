import os
import random
import re
from typing import Dict, List
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from courlan import get_base_url

from src.score.constants import framing_map, persuasion_map


def load_subdomains(file_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.join(current_dir, "labels", file_path)

    with open(absolute_path, "r") as f:
        return set(line.strip().lower() for line in f if line.strip())


def normalize_url_for_framing_and_persuasion(url):
    subdomains = load_subdomains("subdomains-10000.txt")
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    parts = domain.split(".")

    main_domain = domain

    while len(parts) > 2 and parts[0] in subdomains:
        parts.pop(0)
        main_domain = parts[0]

    return main_domain


def get_scores(labels: list, true_label: str):
    scores = []

    for label in labels:
        score = 0
        if true_label == label:
            score = 1
        scores.append({"label": label, "score": score})

    return scores


def normalize_url_candidates(url: str) -> List[str]:
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    candidates = [domain]

    parts = domain.split(".")
    for i in range(len(parts)):
        candidate = ".".join(parts[i:])
        if candidate not in candidates:
            candidates.append(candidate)

    candidates = list(dict.fromkeys(candidates))

    return candidates


def load_factuality_and_bias_scores(base_url: str) -> Dict[str, List[Dict[str, float]]]:
    file_path = "corpus.tsv"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.join(current_dir, "labels", file_path)

    corpus_df = pd.read_csv(absolute_path, delimiter="\t")

    candidates = normalize_url_candidates(base_url)
    matching_rows = pd.DataFrame()
    for candidate in candidates:
        candidate_matches = corpus_df[corpus_df["source_url_normalized"] == candidate]
        if not candidate_matches.empty:
            matching_rows = candidate_matches
            break

    if matching_rows.empty:
        return {"factuality": [], "bias": []}

    most_matching_row = matching_rows.iloc[0]

    return {
        "factuality": get_scores(["low", "mixed", "high"], most_matching_row["fact"]),
        "bias": get_scores(["left", "center", "right"], most_matching_row["bias"]),
    }


def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()


def get_top_k_scores(scores_list, k=5):
    labels = [item["label"] for item in scores_list]
    scores = np.array([item["score"] for item in scores_list])

    probabilities = softmax(scores)

    scored_labels = list(zip(labels, probabilities))
    sorted_scores = sorted(scored_labels, key=lambda x: x[1], reverse=True)

    return [
        {"label": label, "score": float(score)} for label, score in sorted_scores[:k]
    ]


def load_framing_and_persuasion_scores(
    base_url: str,
) -> Dict[str, List[Dict[str, float]]]:
    source = normalize_url_for_framing_and_persuasion(base_url)

    framing_path, persuasion_path = "framing.parquet", "persuasion.parquet"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    framing_abs_path = os.path.join(current_dir, "labels", framing_path)
    persuasion_abs_path = os.path.join(current_dir, "labels", persuasion_path)

    framing_df = pd.read_parquet(framing_abs_path)
    persuasion_df = pd.read_parquet(persuasion_abs_path)

    scores = {"framing": [], "persuasion": []}

    if source in framing_df.index:
        most_matching_framing = framing_df.loc[source]
        framing_scores = [
            {"label": framing_map.get(label), "score": most_matching_framing[label]}
            for label in most_matching_framing.index
        ]
        scores["framing"] = get_top_k_scores(framing_scores, k=5)

    if source in persuasion_df.index:
        most_matching_persuasion = persuasion_df.loc[source]
        persuasion_scores = [
            {
                "label": persuasion_map.get(label),
                "score": most_matching_persuasion[label],
            }
            for label in most_matching_persuasion.index
        ]
        scores["persuasion"] = get_top_k_scores(persuasion_scores, k=5)

    return scores


def load_web_source_scores(base_url: str) -> Dict[str, List[Dict[str, float]]]:
    fact_bias_scores = load_factuality_and_bias_scores(base_url)
    framing_persuasion_scores = load_framing_and_persuasion_scores(base_url)

    return fact_bias_scores | framing_persuasion_scores


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def trim_to_n_words(input_string: str, n: int) -> str:
    words = input_string.split()
    trimmed_words = words[:n]
    return " ".join(trimmed_words)


def add_noise_to_scores(scores, noise_level=0.1):
    noisy_scores = []
    for score_entry in scores:
        original_score = score_entry["score"]

        noise = random.uniform(-noise_level, noise_level)

        noisy_score = max(0, min(1.0, original_score + noise))

        noisy_scores.append({"label": score_entry["label"], "score": noisy_score})

    return noisy_scores


if __name__ == "__main__":
    url = "https://tengrinews.kz/kazakhstan_news/eks-glava-kazavtojola-poluchil-8-let-lisheniya-svobodyi-557239/"
    base_url = get_base_url(url)
    print(base_url)
    scores = load_web_source_scores(base_url)
    print(scores)

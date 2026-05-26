from __future__ import annotations

import numpy as np

from util import aes_sbox, decode_joint_hw_labels, hamming_weight


def build_aes_theoretical_joint_distribution():
    joint_counts = np.zeros((256, 9, 9), dtype=np.int64)
    plaintext_values = np.arange(256, dtype=np.uint8)
    h_m = hamming_weight(plaintext_values).astype(np.int64)
    for key_guess in range(256):
        y_values = aes_sbox(np.bitwise_xor(plaintext_values, key_guess))
        h_y = hamming_weight(y_values).astype(np.int64)
        np.add.at(joint_counts[key_guess], (h_m, h_y), 1)
    joint_probabilities = joint_counts.astype(np.float64) / joint_counts.sum(axis=(1, 2), keepdims=True)
    return joint_counts, joint_probabilities


def build_empirical_joint_distribution(joint_labels, num_bits=8):
    joint_labels = np.asarray(joint_labels, dtype=np.int64).reshape(-1)
    if joint_labels.size == 0:
        raise ValueError("At least one joint label is required.")
    num_hw_classes = num_bits + 1
    h_m, h_y = decode_joint_hw_labels(joint_labels, num_bits=num_bits)
    counts = np.zeros((num_hw_classes, num_hw_classes), dtype=np.int64)
    np.add.at(counts, (h_m, h_y), 1)
    probabilities = counts.astype(np.float64) / joint_labels.size
    return {
        "joint_labels": joint_labels,
        "h_m": h_m,
        "h_y": h_y,
        "hw_pairs": np.column_stack([h_m, h_y]).astype(np.int64),
        "counts": counts,
        "probabilities": probabilities,
    }


def build_empirical_distributions(joint_labels_by_method, num_bits=8):
    return {
        method_name: build_empirical_joint_distribution(joint_labels, num_bits=num_bits)
        for method_name, joint_labels in joint_labels_by_method.items()
    }


def empirical_distribution_summary(distribution):
    probabilities = np.asarray(distribution["probabilities"], dtype=np.float64)
    positive = probabilities[probabilities > 0]
    entropy = -np.sum(positive * np.log(positive))
    normalized_entropy = entropy / np.log(probabilities.size)
    top_h_m, top_h_y = np.unravel_index(int(np.argmax(probabilities)), probabilities.shape)
    return {
        "num_traces": int(distribution["joint_labels"].size),
        "non_empty_cells": int(np.count_nonzero(distribution["counts"])),
        "top_pair": (int(top_h_m), int(top_h_y)),
        "top_probability": float(probabilities[top_h_m, top_h_y]),
        "normalized_entropy": float(normalized_entropy),
    }


def print_empirical_distribution_summary(distributions_by_method):
    for method_name, distribution in distributions_by_method.items():
        summary = empirical_distribution_summary(distribution)
        top_h_m, top_h_y = summary["top_pair"]
        print(method_name)
        print(f"  traces          : {summary['num_traces']}")
        print(f"  non-empty cells : {summary['non_empty_cells']} / 81")
        print(f"  most common pair: (h_m={top_h_m}, h_y={top_h_y}) with p={summary['top_probability']:.3f}")
        print(f"  norm. entropy   : {summary['normalized_entropy']:.3f}")


def smooth_theoretical_probabilities(theoretical_probabilities, epsilon=1e-8):
    theoretical_probabilities = np.asarray(theoretical_probabilities, dtype=np.float64)
    smoothed = np.maximum(theoretical_probabilities, float(epsilon))
    return smoothed / smoothed.sum(axis=(1, 2), keepdims=True)


def score_key_candidates_by_distribution(empirical_probabilities, theoretical_probabilities, epsilon=1e-8):
    empirical_probabilities = np.asarray(empirical_probabilities, dtype=np.float64)
    theoretical_probabilities = smooth_theoretical_probabilities(theoretical_probabilities, epsilon=epsilon)
    log_theoretical = np.log(theoretical_probabilities)
    score = np.sum(empirical_probabilities[np.newaxis, :, :] * log_theoretical, axis=(1, 2))
    ranking = np.argsort(score)[::-1]
    return {
        "score": score,
        "ranking": ranking,
        "best_key": int(ranking[0]),
    }


def estimate_paper_style_noise_variance(trace_matrix, initial_variance=0.001):
    trace_matrix = np.asarray(trace_matrix, dtype=np.float64)
    variances = np.var(trace_matrix, axis=0, dtype=np.float64)
    positive = variances[variances > 0]
    if positive.size == 0:
        return float(initial_variance)
    return float(min(float(initial_variance), float(positive.min())))


def gaussian_density_at_integer_centers(observed_values, centers, std):
    std = max(float(std), 1e-12)
    observed_values = np.asarray(observed_values, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    return (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((observed_values - centers) / std) ** 2)


def build_paper_likelihood_lookup(theoretical_probabilities, std, likelihood_floor=1e-8):
    theoretical_probabilities = np.asarray(theoretical_probabilities, dtype=np.float64)
    num_keys, num_h_m, num_h_y = theoretical_probabilities.shape
    hw_axis = np.arange(num_h_m, dtype=np.float64)
    gaussian_table = gaussian_density_at_integer_centers(
        observed_values=hw_axis[:, np.newaxis],
        centers=hw_axis[np.newaxis, :],
        std=std,
    )
    lookup = np.empty((num_h_m, num_h_y, num_keys), dtype=np.float64)
    for observed_h_m in range(num_h_m):
        for observed_h_y in range(num_h_y):
            cell_likelihood = (
                theoretical_probabilities
                * gaussian_table[observed_h_m][np.newaxis, :, np.newaxis]
                * gaussian_table[observed_h_y][np.newaxis, np.newaxis, :]
            )
            cell_likelihood = np.where(cell_likelihood < likelihood_floor, likelihood_floor, cell_likelihood)
            lookup[observed_h_m, observed_h_y] = cell_likelihood.sum(axis=(1, 2))
    return lookup


def attack_blind_log_numpy(hw_pairs, likelihood_lookup):
    hw_pairs = np.asarray(hw_pairs, dtype=np.int64)
    trace_likelihoods = likelihood_lookup[hw_pairs[:, 0], hw_pairs[:, 1]]
    return np.cumsum(np.log10(trace_likelihoods), axis=0).T


def key_ranks_from_cumulative_scores(cumulative_scores):
    sorted_keys = np.argsort(cumulative_scores, axis=0)[::-1]
    ranks = np.empty_like(sorted_keys)
    ranks[sorted_keys, np.arange(cumulative_scores.shape[1])] = np.arange(cumulative_scores.shape[0])[:, np.newaxis]
    return ranks


def first_trace_count_at_or_below(curve, threshold):
    indices = np.flatnonzero(np.asarray(curve) <= threshold)
    return None if indices.size == 0 else int(indices[0] + 1)


def ntge_from_ge_curve(ge_curve):
    ge_curve = np.asarray(ge_curve)
    ntge = float("inf")
    for index in range(ge_curve.shape[0] - 1, -1, -1):
        if ge_curve[index] > 0:
            break
        if ge_curve[index] == 0:
            ntge = index
    return ntge


def run_paper_style_ge_attack(
    hw_pairs,
    theoretical_probabilities,
    correct_key,
    std,
    num_attacks=100,
    num_traces_attack=1700,
    sr_order=1,
    random_state=2025,
    likelihood_floor=1e-8,
):
    hw_pairs = np.asarray(hw_pairs, dtype=np.int64)
    num_used_traces = min(int(num_traces_attack), hw_pairs.shape[0])
    likelihood_lookup = build_paper_likelihood_lookup(theoretical_probabilities, std, likelihood_floor)
    rng = np.random.default_rng(random_state)
    ge_rounds = np.zeros((int(num_attacks), num_used_traces), dtype=np.float64)
    success_rate_sum = np.zeros(num_used_traces, dtype=np.float64)

    for attack_index in range(int(num_attacks)):
        shuffled_indices = rng.permutation(hw_pairs.shape[0])[:num_used_traces]
        cumulative_scores = attack_blind_log_numpy(hw_pairs[shuffled_indices], likelihood_lookup)
        ranks = key_ranks_from_cumulative_scores(cumulative_scores)
        correct_key_ranks = ranks[int(correct_key)].astype(np.float64)
        ge_rounds[attack_index] = correct_key_ranks
        success_rate_sum += correct_key_ranks <= sr_order

    ge = ge_rounds.mean(axis=0)
    success_rate = success_rate_sum / float(num_attacks)
    fixed_order_scores = attack_blind_log_numpy(hw_pairs[:num_used_traces], likelihood_lookup)
    fixed_order_ranks = key_ranks_from_cumulative_scores(fixed_order_scores)
    final_ranking = np.argsort(fixed_order_scores[:, -1])[::-1]
    return {
        "ge": ge,
        "ge_rounds": ge_rounds,
        "success_rate": success_rate,
        "ntge": ntge_from_ge_curve(ge),
        "trace_count_for_ge_le_sr_order": first_trace_count_at_or_below(ge, sr_order),
        "final_ranking_fixed_order": final_ranking,
        "final_correct_rank_fixed_order": int(fixed_order_ranks[int(correct_key), -1]),
        "num_traces_attack": int(num_used_traces),
        "num_attacks": int(num_attacks),
        "sr_order": int(sr_order),
        "std": float(std),
        "correct_key": int(correct_key),
        "final_ge": float(ge[-1]),
        "final_success_rate": float(success_rate[-1]),
    }


def print_paper_style_ge_summary(results_by_method, top_n=5):
    for method_name, result in results_by_method.items():
        print(method_name)
        print(f"  traces per attack      : {result['num_traces_attack']}")
        print(f"  Gaussian std           : {result['std']:.6f}")
        print(f"  final GE               : {result['final_ge']:.3f}")
        print(f"  final success rate     : {result['final_success_rate']:.3f}")
        print(f"  NTGE, GE=0             : {result['ntge']}")
        print(f"  first trace with GE<=T : {result['trace_count_for_ge_le_sr_order']}")
        print(f"  fixed-order final rank : {result['final_correct_rank_fixed_order']} / 255")
        print("  fixed-order top keys:")
        for rank_index, key_guess in enumerate(result["final_ranking_fixed_order"][:top_n], start=1):
            marker = " <- correct" if int(key_guess) == int(result["correct_key"]) else ""
            print(f"    {rank_index:2d}. key=0x{int(key_guess):02X}{marker}")

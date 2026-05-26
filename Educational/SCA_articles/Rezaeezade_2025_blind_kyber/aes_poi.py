from __future__ import annotations

from pathlib import Path

import numpy as np

from util import compute_aes_hw_pair, ensure_dir


def pearson_correlation_to_trace_samples(traces, target_values):
    traces = np.asarray(traces, dtype=np.float64)
    target_values = np.asarray(target_values, dtype=np.float64).reshape(-1)
    if traces.ndim != 2:
        raise ValueError("traces must have shape (n_traces, n_samples).")
    if traces.shape[0] != target_values.shape[0]:
        raise ValueError("The number of traces must match the number of target values.")

    centered_traces = traces - traces.mean(axis=0, keepdims=True)
    centered_target = target_values - target_values.mean()
    numerator = centered_traces.T @ centered_target
    denominator = np.sqrt(np.sum(centered_traces**2, axis=0) * np.sum(centered_target**2))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def select_top_pois(correlation_trace, num_pois=50, min_separation=1, sort_indices=True):
    correlation_trace = np.asarray(correlation_trace, dtype=np.float64).reshape(-1)
    if num_pois <= 0:
        raise ValueError("num_pois must be positive.")
    if min_separation <= 0:
        raise ValueError("min_separation must be positive.")

    ranked_indices = np.argsort(np.abs(correlation_trace))[::-1]
    selected_indices = []
    for sample_index in ranked_indices:
        sample_index = int(sample_index)
        if all(abs(sample_index - chosen) >= min_separation for chosen in selected_indices):
            selected_indices.append(sample_index)
            if len(selected_indices) == num_pois:
                break
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    return np.sort(selected_indices) if sort_indices else selected_indices


def find_aes_pois_with_correlation(
    traces,
    plaintext_bytes,
    key_bytes,
    byte_index,
    num_pois_per_variable=50,
    min_separation=1,
    sort_indices=True,
):
    traces = np.asarray(traces, dtype=np.float64)
    h_m_target, h_y_target = compute_aes_hw_pair(
        plaintext_bytes=plaintext_bytes,
        key_bytes=key_bytes,
        byte_index=byte_index,
    )
    correlation_h_m = pearson_correlation_to_trace_samples(traces, h_m_target)
    correlation_h_y = pearson_correlation_to_trace_samples(traces, h_y_target)
    pois_h_m = select_top_pois(
        correlation_h_m,
        num_pois=num_pois_per_variable,
        min_separation=min_separation,
        sort_indices=sort_indices,
    )
    pois_h_y = select_top_pois(
        correlation_h_y,
        num_pois=num_pois_per_variable,
        min_separation=min_separation,
        sort_indices=sort_indices,
    )
    return {
        "h_m_target": h_m_target,
        "h_y_target": h_y_target,
        "correlation_h_m": correlation_h_m,
        "correlation_h_y": correlation_h_y,
        "pois_h_m": pois_h_m,
        "pois_h_y": pois_h_y,
    }


def save_poi_result(path, poi_result, metadata=None):
    path = Path(path)
    ensure_dir(path.parent)
    arrays = {key: np.asarray(value) for key, value in poi_result.items()}
    if metadata is not None:
        arrays["metadata"] = np.asarray(metadata, dtype=object)
    np.savez_compressed(path, **arrays)
    return path


def load_poi_result(path):
    data = np.load(path, allow_pickle=True)
    return {
        key: data[key]
        for key in data.files
        if key != "metadata"
    }

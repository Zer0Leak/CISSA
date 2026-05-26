from __future__ import annotations

from math import comb

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from util import compute_aes_hw_pair, encode_joint_hw_labels


def binomial_hw_probabilities(num_bits=8):
    return np.array([comb(num_bits, h) for h in range(num_bits + 1)], dtype=np.float64) / (2**num_bits)


def allocate_slicing_counts(num_items, num_bits=8):
    if (num_items == 81) and (num_bits == 8):
        return np.array([1, 2, 8, 18, 23, 18, 8, 2, 1], dtype=np.int64)
    probabilities = binomial_hw_probabilities(num_bits=num_bits)
    raw_counts = num_items * probabilities
    class_counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(num_items - class_counts.sum())
    if remainder > 0:
        order = np.argsort(raw_counts - class_counts)[::-1]
        class_counts[order[:remainder]] += 1
    return class_counts


def slicing_label_1d(values, num_bits=8):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    class_counts = allocate_slicing_counts(num_items=values.size, num_bits=num_bits)
    sort_order = np.argsort(values)
    labels = np.empty(values.size, dtype=np.int64)
    start = 0
    for hw_label, class_count in enumerate(class_counts):
        if class_count == 0:
            continue
        stop = start + int(class_count)
        labels[sort_order[start:stop]] = hw_label
        start = stop
    return labels, class_counts


def extract_mc_features_from_aes_pois(traces, poi_result, align_signs=True):
    traces = np.asarray(traces, dtype=np.float64)
    pois_h_m = np.asarray(poi_result["pois_h_m"], dtype=np.int64)
    pois_h_y = np.asarray(poi_result["pois_h_y"], dtype=np.int64)
    features_h_m = traces[:, pois_h_m].copy()
    features_h_y = traces[:, pois_h_y].copy()
    sign_h_m = np.sign(np.asarray(poi_result["correlation_h_m"], dtype=np.float64)[pois_h_m])
    sign_h_y = np.sign(np.asarray(poi_result["correlation_h_y"], dtype=np.float64)[pois_h_y])
    sign_h_m[sign_h_m == 0.0] = 1.0
    sign_h_y[sign_h_y == 0.0] = 1.0
    if align_signs:
        features_h_m *= sign_h_m
        features_h_y *= sign_h_y
    return np.concatenate([features_h_m, features_h_y], axis=1), {
        "pois_h_m": pois_h_m,
        "pois_h_y": pois_h_y,
        "sign_h_m": sign_h_m,
        "sign_h_y": sign_h_y,
        "num_pois_h_m": features_h_m.shape[1],
        "num_pois_h_y": features_h_y.shape[1],
    }


def compute_cluster_centers(feature_matrix, cluster_assignments, num_clusters, fallback_centers=None):
    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)
    cluster_assignments = np.asarray(cluster_assignments, dtype=np.int64).reshape(-1)
    cluster_sizes = np.bincount(cluster_assignments, minlength=num_clusters)
    centers = np.zeros((num_clusters, feature_matrix.shape[1]), dtype=np.float64)
    for cluster_index in range(num_clusters):
        in_cluster = cluster_assignments == cluster_index
        if np.any(in_cluster):
            centers[cluster_index] = feature_matrix[in_cluster].mean(axis=0)
        elif fallback_centers is not None:
            centers[cluster_index] = fallback_centers[cluster_index]
    return centers, cluster_sizes


def weighted_majority_vote(candidate_labels, num_bits=8, correction_alpha=0.0):
    candidate_labels = np.asarray(candidate_labels, dtype=np.int64)
    num_classes = num_bits + 1
    slice_counts = allocate_slicing_counts(candidate_labels.shape[0], num_bits=num_bits).astype(np.float64)
    weighted_scores = np.zeros((candidate_labels.shape[0], num_classes), dtype=np.float64)
    for hw_label in range(num_classes):
        vote_count = np.sum(candidate_labels == hw_label, axis=1)
        weight = 1.0 / slice_counts[hw_label]
        correction = (comb(num_bits, hw_label) / slice_counts[hw_label]) ** correction_alpha
        weighted_scores[:, hw_label] = vote_count * weight * correction
    return np.argmax(weighted_scores, axis=1).astype(np.int64), weighted_scores


def expected_binomial_joint_cluster_weights(num_bits=8):
    hw_probabilities = binomial_hw_probabilities(num_bits=num_bits)
    joint_weights = np.outer(hw_probabilities, hw_probabilities).reshape(-1)
    return joint_weights / joint_weights.sum()


def permutation_invariant_size_hinted_weights(
    learned_weights,
    target_weights,
    strength=0.25,
    rank_reference_weights=None,
):
    learned_weights = np.asarray(learned_weights, dtype=np.float64).reshape(-1)
    target_weights = np.asarray(target_weights, dtype=np.float64).reshape(-1)
    if learned_weights.shape != target_weights.shape:
        raise ValueError("learned_weights and target_weights must have the same shape.")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1].")
    epsilon = 1e-12
    learned_weights = np.maximum(learned_weights, epsilon)
    learned_weights /= learned_weights.sum()
    target_weights = np.maximum(target_weights, epsilon)
    target_weights /= target_weights.sum()
    if rank_reference_weights is None:
        rank_reference_weights = learned_weights
    rank_reference_weights = np.asarray(rank_reference_weights, dtype=np.float64).reshape(-1)
    component_order = np.argsort(rank_reference_weights)[::-1]
    sorted_target_weights = np.sort(target_weights)[::-1]
    assigned_target_weights = np.empty_like(target_weights)
    assigned_target_weights[component_order] = sorted_target_weights
    log_hinted = (1.0 - strength) * np.log(learned_weights) + strength * np.log(assigned_target_weights)
    hinted_weights = np.exp(log_hinted - np.max(log_hinted))
    hinted_weights /= hinted_weights.sum()
    return hinted_weights, assigned_target_weights, component_order


def build_mc_labels_from_pois(
    feature_matrix,
    num_pois_h_m,
    num_bits=8,
    covariance_type="full",
    standardize_features=False,
    random_state=0,
    max_iter=100,
    size_hint_strength=0.0,
    size_hint_target_weights=None,
):
    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)
    num_clusters = (num_bits + 1) ** 2
    if standardize_features:
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)

    gmm = GaussianMixture(
        n_components=num_clusters,
        covariance_type=covariance_type,
        reg_covar=1e-6,
        n_init=1,
        max_iter=max_iter,
        random_state=random_state,
    )
    cluster_assignments = gmm.fit_predict(feature_matrix)
    cluster_assignments_before_size_hint = cluster_assignments.copy()
    cluster_sizes_before_size_hint = np.bincount(cluster_assignments, minlength=num_clusters)
    gmm_weights_before_size_hint = gmm.weights_.copy()

    if size_hint_strength > 0.0:
        if size_hint_target_weights is None:
            size_hint_target_weights = expected_binomial_joint_cluster_weights(num_bits=num_bits)
        rank_reference = cluster_sizes_before_size_hint / cluster_sizes_before_size_hint.sum()
        hinted_weights, assigned_target_weights, component_order = permutation_invariant_size_hinted_weights(
            learned_weights=gmm.weights_,
            target_weights=size_hint_target_weights,
            strength=size_hint_strength,
            rank_reference_weights=rank_reference,
        )
        gmm.weights_ = hinted_weights
        cluster_assignments = gmm.predict(feature_matrix)
    else:
        size_hint_target_weights = None
        assigned_target_weights = None
        component_order = None

    cluster_centers, cluster_sizes = compute_cluster_centers(
        feature_matrix,
        cluster_assignments,
        num_clusters,
        fallback_centers=getattr(gmm, "means_", None),
    )
    cluster_centers_h_m = cluster_centers[:, :num_pois_h_m]
    cluster_centers_h_y = cluster_centers[:, num_pois_h_m:]
    candidate_labels_h_m = np.column_stack(
        [slicing_label_1d(cluster_centers_h_m[:, poi_index], num_bits=num_bits)[0] for poi_index in range(cluster_centers_h_m.shape[1])]
    )
    candidate_labels_h_y = np.column_stack(
        [slicing_label_1d(cluster_centers_h_y[:, poi_index], num_bits=num_bits)[0] for poi_index in range(cluster_centers_h_y.shape[1])]
    )
    cluster_labels_h_m, vote_scores_h_m = weighted_majority_vote(candidate_labels_h_m, num_bits=num_bits)
    cluster_labels_h_y, vote_scores_h_y = weighted_majority_vote(candidate_labels_h_y, num_bits=num_bits)
    trace_labels_h_m = cluster_labels_h_m[cluster_assignments]
    trace_labels_h_y = cluster_labels_h_y[cluster_assignments]
    return {
        "feature_matrix": feature_matrix,
        "gmm": gmm,
        "num_clusters": num_clusters,
        "cluster_assignments": cluster_assignments,
        "cluster_assignments_before_size_hint": cluster_assignments_before_size_hint,
        "cluster_sizes": cluster_sizes,
        "cluster_sizes_before_size_hint": cluster_sizes_before_size_hint,
        "cluster_centers": cluster_centers,
        "gmm_weights_before_size_hint": gmm_weights_before_size_hint,
        "size_hint_strength": float(size_hint_strength),
        "size_hint_target_weights": size_hint_target_weights,
        "size_hint_assigned_target_weights": assigned_target_weights,
        "size_hint_component_order": component_order,
        "candidate_labels_h_m": candidate_labels_h_m,
        "candidate_labels_h_y": candidate_labels_h_y,
        "vote_scores_h_m": vote_scores_h_m,
        "vote_scores_h_y": vote_scores_h_y,
        "cluster_labels_h_m": cluster_labels_h_m,
        "cluster_labels_h_y": cluster_labels_h_y,
        "trace_labels_h_m": trace_labels_h_m,
        "trace_labels_h_y": trace_labels_h_y,
    }


def build_mc_label_variants(feature_matrix, num_pois_h_m, variants, size_hint_strength=0.25, **common_kwargs):
    results = {}
    for variant in variants:
        if variant == "original":
            results[variant] = build_mc_labels_from_pois(feature_matrix, num_pois_h_m, size_hint_strength=0.0, **common_kwargs)
        elif variant == "size_hinted":
            results[variant] = build_mc_labels_from_pois(
                feature_matrix,
                num_pois_h_m,
                size_hint_strength=size_hint_strength,
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unsupported MC variant: {variant!r}")
    return results


def mc_accuracy_by_variant(mc_results_by_variant, plaintexts, keys, byte_index=0, num_bits=8):
    true_h_m, true_h_y = compute_aes_hw_pair(plaintexts, keys, byte_index=byte_index)
    true_h_m = true_h_m.astype(np.int64)
    true_h_y = true_h_y.astype(np.int64)
    accuracies = {}
    for variant_name, result in mc_results_by_variant.items():
        correct_h_m = result["trace_labels_h_m"] == true_h_m
        correct_h_y = result["trace_labels_h_y"] == true_h_y
        accuracies[variant_name] = {
            "h_m": float(np.mean(correct_h_m)),
            "h_y": float(np.mean(correct_h_y)),
            "joint": float(np.mean(correct_h_m & correct_h_y)),
        }
    return accuracies, encode_joint_hw_labels(true_h_m, true_h_y, num_bits=num_bits)


def assign_attack_labels_for_mc_variants(mc_results_by_variant, attack_feature_matrix, num_bits=8):
    labels_by_method = {}
    for variant_name, result in mc_results_by_variant.items():
        assignments = result["gmm"].predict(attack_feature_matrix)
        method_name = f"MC {variant_name.replace('_', '-')}"
        labels_by_method[method_name] = encode_joint_hw_labels(
            result["cluster_labels_h_m"][assignments],
            result["cluster_labels_h_y"][assignments],
            num_bits=num_bits,
        )
    return labels_by_method

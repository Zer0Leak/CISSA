from __future__ import annotations

from math import comb

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import (
    _compute_precision_cholesky,
    _estimate_gaussian_parameters,
    _estimate_log_gaussian_prob,
)
from sklearn.preprocessing import StandardScaler

from util import compute_aes_hw_pair, encode_joint_hw_labels


class FixedPriorGaussianMixture(GaussianMixture):
    """Soft-EM GMM variant that keeps mixture priors fixed during every M-step."""

    def __init__(
        self,
        n_components=1,
        *,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
        fixed_weights=None,
        feedback_strength=0.0,
        initial_effective_weights=None,
    ):
        self.fixed_weights = fixed_weights
        self.feedback_strength = feedback_strength
        self.initial_effective_weights = initial_effective_weights
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            weights_init=fixed_weights if fixed_weights is not None else weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )

    def _initialize(self, X, resp, xp=None):
        super()._initialize(X, resp, xp=xp)
        if self.fixed_weights is not None:
            fixed_weights = np.asarray(self.fixed_weights, dtype=np.float64)
            self.weights_ = fixed_weights / fixed_weights.sum()
            if self.initial_effective_weights is None:
                self.effective_weights_ = self.weights_.copy()
            else:
                effective_weights = np.asarray(self.initial_effective_weights, dtype=np.float64)
                effective_weights = np.maximum(effective_weights, 1e-300)
                self.effective_weights_ = effective_weights / effective_weights.sum()

    def _estimate_log_weights(self, xp=None):
        if self.fixed_weights is None:
            return super()._estimate_log_weights(xp=xp)
        fixed_weights = np.asarray(self.fixed_weights, dtype=np.float64)
        fixed_weights = np.maximum(fixed_weights / fixed_weights.sum(), 1e-300)
        log_weights = np.log(fixed_weights)
        feedback_strength = float(self.feedback_strength)
        if feedback_strength > 0.0:
            effective_weights = np.asarray(getattr(self, "effective_weights_", fixed_weights), dtype=np.float64)
            effective_weights = np.maximum(effective_weights / effective_weights.sum(), 1e-300)
            log_weights = log_weights + feedback_strength * (log_weights - np.log(effective_weights))
        return log_weights

    def _m_step(self, X, log_resp, xp=None):
        if self.fixed_weights is None:
            super()._m_step(X, log_resp, xp=xp)
            return
        resp = np.exp(log_resp)
        nk, self.means_, self.covariances_ = _estimate_gaussian_parameters(
            X,
            resp,
            self.reg_covar,
            self.covariance_type,
            xp=xp,
        )
        effective_weights = np.asarray(nk, dtype=np.float64)
        effective_weights = np.maximum(effective_weights, 1e-300)
        self.effective_weights_ = effective_weights / effective_weights.sum()
        fixed_weights = np.asarray(self.fixed_weights, dtype=np.float64)
        self.weights_ = fixed_weights / fixed_weights.sum()
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_,
            self.covariance_type,
            xp=xp,
        )


class HardSizeConstrainedGaussianMixture:
    """Hard-EM GMM variant that enforces fixed component counts on training data."""

    def __init__(
        self,
        component_counts,
        *,
        covariance_type="full",
        reg_covar=1e-6,
        max_iter=100,
        tol=1e-4,
        means_init=None,
        covariances_init=None,
    ):
        if covariance_type != "full":
            raise ValueError("HardSizeConstrainedGaussianMixture currently supports covariance_type='full' only.")
        self.component_counts = np.asarray(component_counts, dtype=np.int64)
        if np.any(self.component_counts < 0):
            raise ValueError("component_counts must be non-negative.")
        self.n_components = int(self.component_counts.size)
        self.covariance_type = covariance_type
        self.reg_covar = float(reg_covar)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.means_init = means_init
        self.covariances_init = covariances_init

    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X,
            self.means_,
            self.precisions_cholesky_,
            self.covariance_type,
        )

    def _estimate_weighted_log_prob(self, X):
        log_weights = np.full(self.n_components, -np.inf, dtype=np.float64)
        positive = self.weights_ > 0.0
        log_weights[positive] = np.log(self.weights_[positive])
        return self._estimate_log_prob(X) + log_weights[np.newaxis, :]

    def _update_parameters_from_hard_assignments(self, X, labels):
        n_features = X.shape[1]
        covariances = np.empty((self.n_components, n_features, n_features), dtype=np.float64)
        means = np.asarray(self.means_, dtype=np.float64).copy()
        for component_index in range(self.n_components):
            members = X[labels == component_index]
            if members.shape[0] > 0:
                means[component_index] = members.mean(axis=0)
                diff = members - means[component_index]
                covariances[component_index] = (diff.T @ diff) / float(members.shape[0])
                covariances[component_index].flat[:: n_features + 1] += self.reg_covar
            else:
                covariances[component_index] = self.covariances_[component_index]
        self.means_ = means
        self.covariances_ = covariances
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, self.covariance_type)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.component_counts.sum() != X.shape[0]:
            raise ValueError("component_counts must sum to the number of training samples.")
        if self.means_init is None or self.covariances_init is None:
            initial_gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                n_init=1,
                max_iter=self.max_iter,
                random_state=0,
            )
            initial_gmm.fit(X)
            self.means_ = initial_gmm.means_.copy()
            self.covariances_ = initial_gmm.covariances_.copy()
        else:
            self.means_ = np.asarray(self.means_init, dtype=np.float64).copy()
            self.covariances_ = np.asarray(self.covariances_init, dtype=np.float64).copy()
        self.weights_ = self.component_counts.astype(np.float64) / float(X.shape[0])
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, self.covariance_type)

        previous_score = -np.inf
        previous_labels = None
        for iteration in range(self.max_iter):
            scores = self._estimate_log_prob(X)
            labels = constrained_hard_assignment(scores, self.component_counts)
            selected_score = float(np.mean(scores[np.arange(X.shape[0]), labels]))
            self._update_parameters_from_hard_assignments(X, labels)
            if previous_labels is not None:
                assignment_change = float(np.mean(labels != previous_labels))
                if assignment_change == 0.0 or abs(selected_score - previous_score) < self.tol:
                    break
            previous_score = selected_score
            previous_labels = labels.copy()

        self.labels_ = labels
        self.n_iter_ = iteration + 1
        self.lower_bound_ = selected_score
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = np.max(weighted_log_prob, axis=1, keepdims=True)
        stabilized = np.exp(weighted_log_prob - log_prob_norm)
        row_sums = stabilized.sum(axis=1, keepdims=True)
        return np.divide(stabilized, row_sums, out=np.zeros_like(stabilized), where=row_sums > 0.0)

    def predict(self, X):
        return np.argmax(self._estimate_weighted_log_prob(X), axis=1).astype(np.int64)


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


def label_cluster_centers(cluster_centers, num_pois_h_m, num_bits=8):
    cluster_centers = np.asarray(cluster_centers, dtype=np.float64)
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
    return {
        "candidate_labels_h_m": candidate_labels_h_m,
        "candidate_labels_h_y": candidate_labels_h_y,
        "vote_scores_h_m": vote_scores_h_m,
        "vote_scores_h_y": vote_scores_h_y,
        "cluster_labels_h_m": cluster_labels_h_m,
        "cluster_labels_h_y": cluster_labels_h_y,
    }


def expected_binomial_joint_cluster_weights(num_bits=8):
    hw_probabilities = binomial_hw_probabilities(num_bits=num_bits)
    joint_weights = np.outer(hw_probabilities, hw_probabilities).reshape(-1)
    return joint_weights / joint_weights.sum()


def expected_binomial_joint_cluster_sizes(num_traces, num_bits=8):
    return expected_binomial_joint_cluster_weights(num_bits=num_bits) * float(num_traces)


def integer_counts_from_probabilities(probabilities, num_items):
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    probabilities = probabilities / probabilities.sum()
    raw_counts = probabilities * int(num_items)
    counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(num_items - counts.sum())
    if remainder > 0:
        order = np.argsort(raw_counts - counts)[::-1]
        counts[order[:remainder]] += 1
    return counts


def precision_init_from_covariances(covariances, covariance_type):
    covariances = np.asarray(covariances, dtype=np.float64)
    if covariance_type == "full":
        return np.linalg.inv(covariances)
    if covariance_type == "tied":
        return np.linalg.inv(covariances)
    if covariance_type in {"diag", "spherical"}:
        return 1.0 / covariances
    raise ValueError(f"Unsupported covariance_type: {covariance_type!r}")


def constrained_hard_assignment(scores, component_counts):
    """Assign every sample to one component while exactly filling component_counts."""

    scores = np.asarray(scores, dtype=np.float64)
    component_counts = np.asarray(component_counts, dtype=np.int64).reshape(-1)
    num_samples, num_components = scores.shape
    if component_counts.size != num_components:
        raise ValueError("component_counts must have one count per score column.")
    if int(component_counts.sum()) != num_samples:
        raise ValueError("component_counts must sum to the number of samples.")

    labels = np.full(num_samples, -1, dtype=np.int64)
    remaining = component_counts.copy()
    flat_order = np.argsort(scores.ravel())[::-1]
    for flat_index in flat_order:
        sample_index = int(flat_index // num_components)
        component_index = int(flat_index % num_components)
        if labels[sample_index] >= 0 or remaining[component_index] <= 0:
            continue
        labels[sample_index] = component_index
        remaining[component_index] -= 1
        if np.all(labels >= 0):
            break

    if np.any(labels < 0):
        available_components = np.repeat(np.arange(num_components, dtype=np.int64), remaining)
        labels[labels < 0] = available_components[: np.sum(labels < 0)]
    return labels


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


def marginal_size_hinted_weights(
    learned_weights,
    cluster_labels_h_m,
    cluster_labels_h_y,
    num_bits=8,
    strength=1.0,
    max_iter=200,
    tolerance=1e-12,
):
    learned_weights = np.asarray(learned_weights, dtype=np.float64).reshape(-1)
    cluster_labels_h_m = np.asarray(cluster_labels_h_m, dtype=np.int64).reshape(-1)
    cluster_labels_h_y = np.asarray(cluster_labels_h_y, dtype=np.int64).reshape(-1)
    if learned_weights.shape != cluster_labels_h_m.shape or learned_weights.shape != cluster_labels_h_y.shape:
        raise ValueError("learned_weights and cluster labels must have the same length.")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1].")

    epsilon = 1e-300
    learned_weights = np.maximum(learned_weights, epsilon)
    learned_weights /= learned_weights.sum()
    adjusted_weights = learned_weights.copy()
    target_marginal = binomial_hw_probabilities(num_bits=num_bits)

    for _ in range(int(max_iter)):
        previous_weights = adjusted_weights.copy()
        for labels in (cluster_labels_h_m, cluster_labels_h_y):
            for hw_label, target_probability in enumerate(target_marginal):
                in_label = labels == hw_label
                current_probability = adjusted_weights[in_label].sum()
                if current_probability > 0.0:
                    adjusted_weights[in_label] *= target_probability / current_probability
        adjusted_weights = np.maximum(adjusted_weights, epsilon)
        adjusted_weights /= adjusted_weights.sum()
        if np.max(np.abs(adjusted_weights - previous_weights)) < tolerance:
            break

    hinted_log_weights = (1.0 - strength) * np.log(learned_weights) + strength * np.log(adjusted_weights)
    hinted_weights = np.exp(hinted_log_weights - np.max(hinted_log_weights))
    hinted_weights /= hinted_weights.sum()
    return hinted_weights, adjusted_weights, {
        "target_marginal": target_marginal,
        "prior_labels_h_m": cluster_labels_h_m,
        "prior_labels_h_y": cluster_labels_h_y,
    }


def build_mc_labels_from_pois(
    feature_matrix,
    num_pois_h_m,
    num_bits=8,
    covariance_type="full",
    standardize_features=False,
    random_state=0,
    max_iter=100,
    size_hint_strength=0.0,
    size_hint_mode="none",
    soft_size_feedback_strength=0.0,
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
    initial_cluster_centers, _ = compute_cluster_centers(
        feature_matrix,
        cluster_assignments_before_size_hint,
        num_clusters,
        fallback_centers=getattr(gmm, "means_", None),
    )
    initial_label_info = label_cluster_centers(initial_cluster_centers, num_pois_h_m, num_bits=num_bits)
    fixed_prior_weights = None
    assigned_target_weights = None
    component_order = None
    target_counts = None
    size_hint_target_marginal = None
    size_hint_prior_labels_h_m = None
    size_hint_prior_labels_h_y = None

    if size_hint_mode == "none" or size_hint_strength <= 0.0:
        size_hint_target_weights = None
        size_hint_mode = "none"
    elif size_hint_mode in {"soft_size", "soft_fixed_prior"}:
        rank_reference = cluster_sizes_before_size_hint / cluster_sizes_before_size_hint.sum()
        fixed_prior_weights, assigned_target_weights, marginal_hint_info = marginal_size_hinted_weights(
            learned_weights=gmm.weights_,
            cluster_labels_h_m=initial_label_info["cluster_labels_h_m"],
            cluster_labels_h_y=initial_label_info["cluster_labels_h_y"],
            num_bits=num_bits,
            strength=size_hint_strength,
        )
        size_hint_target_marginal = marginal_hint_info["target_marginal"]
        size_hint_prior_labels_h_m = marginal_hint_info["prior_labels_h_m"]
        size_hint_prior_labels_h_y = marginal_hint_info["prior_labels_h_y"]
        fixed_prior_gmm = FixedPriorGaussianMixture(
            n_components=num_clusters,
            covariance_type=covariance_type,
            reg_covar=1e-6,
            n_init=1,
            max_iter=max_iter,
            random_state=random_state,
            fixed_weights=fixed_prior_weights,
            feedback_strength=soft_size_feedback_strength,
            initial_effective_weights=rank_reference,
            means_init=gmm.means_,
            precisions_init=precision_init_from_covariances(gmm.covariances_, covariance_type),
        )
        fixed_prior_gmm.fit(feature_matrix)
        gmm = fixed_prior_gmm
        cluster_assignments = gmm.predict(feature_matrix)
        size_hint_mode = "soft_fixed_prior_em"
    elif size_hint_mode in {"hard_size", "hard_size_em"}:
        fixed_prior_weights, assigned_target_weights, marginal_hint_info = marginal_size_hinted_weights(
            learned_weights=gmm.weights_,
            cluster_labels_h_m=initial_label_info["cluster_labels_h_m"],
            cluster_labels_h_y=initial_label_info["cluster_labels_h_y"],
            num_bits=num_bits,
            strength=size_hint_strength,
        )
        size_hint_target_marginal = marginal_hint_info["target_marginal"]
        size_hint_prior_labels_h_m = marginal_hint_info["prior_labels_h_m"]
        size_hint_prior_labels_h_y = marginal_hint_info["prior_labels_h_y"]
        target_counts = integer_counts_from_probabilities(fixed_prior_weights, feature_matrix.shape[0])
        hard_size_gmm = HardSizeConstrainedGaussianMixture(
            component_counts=target_counts,
            covariance_type=covariance_type,
            reg_covar=1e-6,
            max_iter=max_iter,
            means_init=gmm.means_,
            covariances_init=gmm.covariances_,
        )
        hard_size_gmm.fit(feature_matrix)
        gmm = hard_size_gmm
        cluster_assignments = gmm.labels_
        size_hint_mode = "hard_size_em"
    else:
        raise ValueError(f"Unsupported size_hint_mode: {size_hint_mode!r}")

    cluster_centers, cluster_sizes = compute_cluster_centers(
        feature_matrix,
        cluster_assignments,
        num_clusters,
        fallback_centers=getattr(gmm, "means_", None),
    )
    final_label_info = label_cluster_centers(cluster_centers, num_pois_h_m, num_bits=num_bits)
    candidate_labels_h_m = final_label_info["candidate_labels_h_m"]
    candidate_labels_h_y = final_label_info["candidate_labels_h_y"]
    vote_scores_h_m = final_label_info["vote_scores_h_m"]
    vote_scores_h_y = final_label_info["vote_scores_h_y"]
    cluster_labels_h_m = final_label_info["cluster_labels_h_m"]
    cluster_labels_h_y = final_label_info["cluster_labels_h_y"]
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
        "size_hint_mode": size_hint_mode,
        "soft_size_feedback_strength": float(soft_size_feedback_strength),
        "size_hint_target_weights": size_hint_target_weights,
        "size_hint_target_marginal": size_hint_target_marginal,
        "size_hint_assigned_target_weights": assigned_target_weights,
        "size_hint_fixed_prior_weights": fixed_prior_weights,
        "size_hint_target_counts": target_counts,
        "size_hint_prior_labels_h_m": size_hint_prior_labels_h_m,
        "size_hint_prior_labels_h_y": size_hint_prior_labels_h_y,
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


def build_mc_label_variants(
    feature_matrix,
    num_pois_h_m,
    variants,
    size_hint_strength=0.25,
    soft_size_feedback_strength=0.0,
    **common_kwargs,
):
    results = {}
    for variant in variants:
        if variant == "original":
            results[variant] = build_mc_labels_from_pois(
                feature_matrix,
                num_pois_h_m,
                size_hint_strength=0.0,
                size_hint_mode="none",
                **common_kwargs,
            )
        elif variant in {"soft_size", "soft_hinted", "size_hinted"}:
            results[variant] = build_mc_labels_from_pois(
                feature_matrix,
                num_pois_h_m,
                size_hint_strength=size_hint_strength,
                size_hint_mode="soft_size",
                soft_size_feedback_strength=soft_size_feedback_strength,
                **common_kwargs,
            )
        elif variant in {"hard_size", "hard_hinted"}:
            results[variant] = build_mc_labels_from_pois(
                feature_matrix,
                num_pois_h_m,
                size_hint_strength=size_hint_strength,
                size_hint_mode="hard_size",
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

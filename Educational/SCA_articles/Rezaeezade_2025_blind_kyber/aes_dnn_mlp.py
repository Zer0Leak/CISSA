from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from util import encode_joint_hw_labels, train_validation_indices


def require_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
        from tensorflow.keras.losses import CategoricalCrossentropy
        from tensorflow.keras.metrics import CategoricalAccuracy
        from tensorflow.keras.optimizers import Adam
    except ImportError as exc:
        raise ImportError("TensorFlow is required for DNN training/prediction.") from exc
    return {
        "tf": tf,
        "Sequential": Sequential,
        "EarlyStopping": EarlyStopping,
        "ReduceLROnPlateau": ReduceLROnPlateau,
        "BatchNormalization": BatchNormalization,
        "Dense": Dense,
        "Dropout": Dropout,
        "Input": Input,
        "CategoricalCrossentropy": CategoricalCrossentropy,
        "CategoricalAccuracy": CategoricalAccuracy,
        "Adam": Adam,
    }


def configure_tensorflow_runtime():
    deps = require_tensorflow()
    tf = deps["tf"]
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        training_device = "/GPU:0"
    else:
        training_device = "/CPU:0"
    print("TensorFlow version:", tf.__version__)
    print("Training device requested:", training_device)
    return gpus, training_device


def one_hot(labels, num_classes):
    labels = np.asarray(labels, dtype=np.int64)
    targets = np.zeros((labels.size, int(num_classes)), dtype=np.float32)
    targets[np.arange(labels.size), labels] = 1.0
    return targets


def candidate_label_vote_counts(candidate_labels, num_classes):
    candidate_labels = np.asarray(candidate_labels, dtype=np.int64)
    counts = np.zeros((candidate_labels.shape[0], num_classes), dtype=np.int64)
    for class_index in range(num_classes):
        counts[:, class_index] = np.sum(candidate_labels == class_index, axis=1)
    return counts


def vote_distribution_confidence(candidate_labels, num_classes):
    vote_counts = candidate_label_vote_counts(candidate_labels, num_classes)
    total_votes = vote_counts.sum(axis=1, keepdims=True)
    vote_probabilities = np.divide(vote_counts, total_votes, out=np.zeros_like(vote_counts, dtype=np.float64), where=total_votes > 0)
    log_vote_probabilities = np.where(vote_probabilities > 0, np.log(vote_probabilities), 0.0)
    entropy = -np.sum(vote_probabilities * log_vote_probabilities, axis=1)
    entropy_confidence = 1.0 - entropy / np.log(num_classes)
    sorted_probabilities = np.sort(vote_probabilities, axis=1)
    margin_confidence = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    return 0.75 * entropy_confidence + 0.25 * margin_confidence


def build_mc_confidence_weights(mc_result, min_weight=0.25, max_weight=1.25):
    num_classes = 9
    confidence_h_m = vote_distribution_confidence(mc_result["candidate_labels_h_m"], num_classes)
    confidence_h_y = vote_distribution_confidence(mc_result["candidate_labels_h_y"], num_classes)
    cluster_confidence = 0.5 * (confidence_h_m + confidence_h_y)
    cluster_weights = 0.25 + 0.75 * cluster_confidence
    trace_confidence = cluster_confidence[mc_result["cluster_assignments"]]
    trace_weights = cluster_weights[mc_result["cluster_assignments"]]
    trace_weights = np.clip(trace_weights, min_weight, max_weight)
    trace_weights = trace_weights / np.mean(trace_weights)
    return trace_weights.astype(np.float32), trace_confidence.astype(np.float64), {
        "confidence_h_m": confidence_h_m,
        "confidence_h_y": confidence_h_y,
        "vote_confidence": cluster_confidence,
        "cluster_weights": cluster_weights,
    }


def prepare_dnn_training_data(
    mc_traces,
    mc_result,
    validation_fraction=0.2,
    random_state=2025,
    num_bits=8,
):
    num_joint_classes = (num_bits + 1) ** 2
    pseudo_joint_labels = encode_joint_hw_labels(
        mc_result["trace_labels_h_m"],
        mc_result["trace_labels_h_y"],
        num_bits=num_bits,
    )
    sample_weights, trace_confidence, cluster_confidence = build_mc_confidence_weights(mc_result)
    train_indices, validation_indices = train_validation_indices(
        labels=pseudo_joint_labels,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    scaler = StandardScaler()
    train_features = scaler.fit_transform(mc_traces[train_indices]).astype(np.float32)
    validation_features = scaler.transform(mc_traces[validation_indices]).astype(np.float32)
    train_labels = pseudo_joint_labels[train_indices]
    validation_labels = pseudo_joint_labels[validation_indices]
    return {
        "scaler": scaler,
        "pseudo_joint_labels": pseudo_joint_labels,
        "sample_weights": sample_weights,
        "trace_confidence": trace_confidence,
        "cluster_confidence": cluster_confidence,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "train_features": train_features,
        "validation_features": validation_features,
        "train_labels": train_labels,
        "validation_labels": validation_labels,
        "train_targets": one_hot(train_labels, num_joint_classes),
        "validation_targets": one_hot(validation_labels, num_joint_classes),
        "train_weights": sample_weights[train_indices].astype(np.float32),
        "num_joint_classes": num_joint_classes,
    }


def build_keras_mlp_like_original_code(classes, number_of_samples, m_params, dropout=True, label_smoothing=0.05):
    deps = require_tensorflow()
    tf = deps["tf"]
    tf.keras.utils.set_random_seed(int(m_params["seed"]))
    model = deps["Sequential"](name="mc_label_mlp_original_style")
    model.add(deps["Input"](shape=(number_of_samples,), name="full_trace"))
    model.add(deps["BatchNormalization"](name="input_batch_normalization"))
    for layer_index in range(int(m_params["layers"])):
        model.add(
            deps["Dense"](
                int(m_params["neurons"]),
                activation=m_params["activation"],
                kernel_initializer=m_params.get("kernel_initializer", "he_uniform"),
                bias_initializer="zeros",
                name=f"dense_{layer_index + 1}",
            )
        )
        if dropout:
            model.add(deps["Dropout"](0.2, name=f"dropout_{layer_index + 1}"))
    model.add(deps["Dense"](classes, activation="softmax", name="joint_hw_softmax"))
    model.compile(
        optimizer=deps["Adam"](learning_rate=float(m_params["learning_rate"])),
        loss=deps["CategoricalCrossentropy"](label_smoothing=label_smoothing),
        metrics=[deps["CategoricalAccuracy"](name="accuracy")],
    )
    return model


def make_weighted_tf_dataset(features, targets, sample_weights, batch_size, shuffle=True, random_state=0):
    deps = require_tensorflow()
    tf = deps["tf"]
    dataset = tf.data.Dataset.from_tensor_slices((features, targets, sample_weights))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=int(features.shape[0]), seed=int(random_state), reshuffle_each_iteration=True)
    return dataset.batch(int(batch_size)).prefetch(tf.data.AUTOTUNE)


def train_mlp(dnn_data, params, epochs=100, dropout=True, label_smoothing=0.05, training_device="/CPU:0"):
    deps = require_tensorflow()
    tf = deps["tf"]
    train_dataset = make_weighted_tf_dataset(
        dnn_data["train_features"],
        dnn_data["train_targets"],
        dnn_data["train_weights"],
        batch_size=params["mini_batch"],
        shuffle=True,
        random_state=params["seed"],
    )
    callbacks = [
        deps["EarlyStopping"](monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        deps["ReduceLROnPlateau"](monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
    ]
    with tf.device(training_device):
        model = build_keras_mlp_like_original_code(
            classes=dnn_data["num_joint_classes"],
            number_of_samples=dnn_data["train_features"].shape[1],
            m_params=params,
            dropout=dropout,
            label_smoothing=label_smoothing,
        )
        history = model.fit(
            train_dataset,
            validation_data=(dnn_data["validation_features"], dnn_data["validation_targets"]),
            epochs=int(epochs),
            verbose=1,
            callbacks=callbacks,
        )
    return model, history


def predict_mlp(model, traces, scaler, batch_size=256):
    features = scaler.transform(traces).astype(np.float32)
    probabilities = model.predict(features, batch_size=int(batch_size), verbose=0)
    return probabilities, np.argmax(probabilities, axis=1), features

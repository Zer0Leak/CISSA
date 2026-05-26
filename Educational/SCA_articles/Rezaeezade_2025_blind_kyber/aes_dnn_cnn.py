from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aes_dnn_mlp import make_weighted_tf_dataset, require_tensorflow
from util import ensure_dir, make_json_serializable


def make_pooling_layer(pooling_type, pool_size, stride, name):
    deps = require_tensorflow()
    from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D

    pooling_type = str(pooling_type).lower()
    if pooling_type.startswith("avg") or pooling_type.startswith("average"):
        return AveragePooling1D(pool_size=int(pool_size), strides=int(stride), padding="same", name=name)
    if pooling_type.startswith("max"):
        return MaxPooling1D(pool_size=int(pool_size), strides=int(stride), padding="same", name=name)
    raise ValueError(f"Unsupported pooling_type: {pooling_type!r}")


def build_keras_cnn_like_original_code(classes, number_of_samples, c_params, dropout=True, label_smoothing=0.05):
    deps = require_tensorflow()
    tf = deps["tf"]
    from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Flatten, Input

    tf.keras.utils.set_random_seed(int(c_params["seed"]))
    model = deps["Sequential"](name="mc_label_cnn_original_style")
    model.add(Input(shape=(number_of_samples, 1), name="trace_sequence"))
    for layer_index in range(int(c_params["conv_layers"])):
        model.add(
            Conv1D(
                filters=int(c_params["filters"][layer_index]),
                kernel_size=int(c_params["kernel_size"][layer_index]),
                strides=int(c_params["strides"][layer_index]),
                padding="same",
                activation=c_params["activation"],
                kernel_initializer=c_params.get("kernel_initializer", "he_uniform"),
                bias_initializer="zeros",
                name=f"conv_{layer_index + 1}",
            )
        )
        model.add(
            make_pooling_layer(
                pooling_type=c_params.get("pooling_type", "Average"),
                pool_size=c_params["pooling_sizes"][layer_index],
                stride=c_params["pooling_strides"][layer_index],
                name=f"pool_{layer_index + 1}",
            )
        )
        model.add(BatchNormalization(name=f"batch_normalization_{layer_index + 1}"))
    model.add(Flatten(name="flatten"))
    for dense_index in range(int(c_params["dense_layers"])):
        model.add(
            Dense(
                int(c_params["neurons"]),
                activation=c_params["activation"],
                kernel_initializer=c_params.get("kernel_initializer", "he_uniform"),
                bias_initializer="zeros",
                name=f"dense_{dense_index + 1}",
            )
        )
        if dropout:
            model.add(Dropout(0.5, name=f"dropout_{dense_index + 1}"))
    model.add(Dense(classes, activation="softmax", name="joint_hw_softmax"))
    model.compile(
        optimizer=deps["Adam"](learning_rate=float(c_params["learning_rate"])),
        loss=deps["CategoricalCrossentropy"](label_smoothing=label_smoothing),
        metrics=[deps["CategoricalAccuracy"](name="accuracy")],
    )
    return model


def sample_cnn_random_search_params(rng, seed):
    conv_layers = int(rng.choice([2, 3, 4]))
    first_filters = int(rng.choice([4, 8, 12, 16, 24]))
    filters = [first_filters]
    for _ in range(1, conv_layers):
        filters.append(filters[-1] * 2)
    pooling_type = str(rng.choice(["Average", "Max"]))
    return {
        "seed": int(seed),
        "mini_batch": int(rng.choice([128, 256, 512])),
        "learning_rate": float(rng.choice([1e-3, 5e-4, 1e-4, 5e-5, 1e-5])),
        "activation": str(rng.choice(["relu", "selu", "elu", "tanh"])),
        "kernel_initializer": str(rng.choice(["random_uniform", "glorot_uniform", "he_uniform"])),
        "dense_layers": int(rng.choice([2, 3])),
        "neurons": int(rng.choice([50, 100, 150, 200, 300, 400, 500])),
        "conv_layers": conv_layers,
        "pooling_type": pooling_type,
        "pooling_types": [pooling_type] * conv_layers,
        "filters": filters,
        "kernel_size": [int(rng.choice(np.arange(4, 20, 2))) for _ in range(conv_layers)],
        "strides": [int(rng.choice([2, 4, 6, 8, 10])) for _ in range(conv_layers)],
        "pooling_sizes": [int(rng.choice([4, 6, 8, 10])) for _ in range(conv_layers)],
        "pooling_strides": [int(rng.choice([4, 6, 8, 10])) for _ in range(conv_layers)],
    }


def summarize_cnn_search_params(params):
    return (
        f"batch={params['mini_batch']}, lr={params['learning_rate']:.0e}, "
        f"act={params['activation']}, conv={params['conv_layers']}, "
        f"filters={params['filters']}, dense={params['dense_layers']}x{params['neurons']}"
    )


def train_cnn_random_search(
    dnn_data,
    trials=12,
    epochs=100,
    seed=2026,
    dropout=True,
    label_smoothing=0.05,
    training_device="/CPU:0",
    output_dir=None,
):
    deps = require_tensorflow()
    tf = deps["tf"]
    rng = np.random.default_rng(seed)
    train_features = np.expand_dims(dnn_data["train_features"], axis=-1)
    validation_features = np.expand_dims(dnn_data["validation_features"], axis=-1)
    search_results = []
    best_trial = None
    best_weights = None
    for trial_index in range(int(trials)):
        trial_seed = int(seed) + trial_index
        params = sample_cnn_random_search_params(rng, seed=trial_seed)
        print(f"\nCNN trial {trial_index + 1}/{trials}: {summarize_cnn_search_params(params)}")
        train_dataset = make_weighted_tf_dataset(
            train_features,
            dnn_data["train_targets"],
            dnn_data["train_weights"],
            batch_size=params["mini_batch"],
            shuffle=True,
            random_state=params["seed"],
        )
        callbacks = [
            deps["EarlyStopping"](monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
            deps["ReduceLROnPlateau"](monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=0),
        ]
        with tf.device(training_device):
            trial_model = build_keras_cnn_like_original_code(
                classes=dnn_data["num_joint_classes"],
                number_of_samples=dnn_data["train_features"].shape[1],
                c_params=params,
                dropout=dropout,
                label_smoothing=label_smoothing,
            )
            history = trial_model.fit(
                train_dataset,
                validation_data=(validation_features, dnn_data["validation_targets"]),
                epochs=int(epochs),
                verbose=0,
                callbacks=callbacks,
            )
        best_val_loss = float(np.min(history.history["val_loss"]))
        record = {
            "trial": trial_index + 1,
            "params": params,
            "history": history.history,
            "best_val_loss": best_val_loss,
            "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
            "final_val_accuracy": float(history.history.get("val_accuracy", [np.nan])[-1]),
        }
        search_results.append(record)
        print(f"  best val_loss={record['best_val_loss']:.4f} at epoch {record['best_epoch']}")
        if best_trial is None or best_val_loss < best_trial["best_val_loss"]:
            best_trial = record
            best_weights = trial_model.get_weights()
        del trial_model
        tf.keras.backend.clear_session()

    params = best_trial["params"]
    with tf.device(training_device):
        model = build_keras_cnn_like_original_code(
            classes=dnn_data["num_joint_classes"],
            number_of_samples=dnn_data["train_features"].shape[1],
            c_params=params,
            dropout=dropout,
            label_smoothing=label_smoothing,
        )
        model.set_weights(best_weights)

    export = {
        "best_trial": int(best_trial["trial"]),
        "best_val_loss": float(best_trial["best_val_loss"]),
        "best_epoch": int(best_trial["best_epoch"]),
        "best_params": make_json_serializable(params),
        "random_search_trials": int(trials),
        "random_search_seed": int(seed),
        "all_trials": make_json_serializable(search_results),
    }
    if output_dir is not None:
        output_dir = ensure_dir(output_dir)
        with open(Path(output_dir) / "cnn_random_search_best_params.json", "w", encoding="utf-8") as file:
            json.dump(export, file, indent=2)
        np.save(Path(output_dir) / "cnn_random_search_best_params.npy", export, allow_pickle=True)
        model.save_weights(str(Path(output_dir) / "cnn_random_search_best_model.weights.h5"))
    return model, best_trial, search_results, export


def predict_cnn(model, traces, scaler, batch_size=256):
    features = scaler.transform(traces).astype(np.float32)
    features_tf = np.expand_dims(features, axis=-1)
    probabilities = model.predict(features_tf, batch_size=int(batch_size), verbose=0)
    return probabilities, np.argmax(probabilities, axis=1), features

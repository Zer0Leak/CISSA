from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


AES_SBOX = np.array(
    [
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
    ],
    dtype=np.uint8,
)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(path: str | Path, obj) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as file:
        pickle.dump(obj, file)
    return path


def load_pickle(path: str | Path):
    with open(path, "rb") as file:
        return pickle.load(file)


def save_json(path: str | Path, obj) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(make_json_serializable(obj), file, indent=2)
    return path


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def make_json_serializable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    return value


def aes_sbox(values):
    return AES_SBOX[np.asarray(values, dtype=np.uint8)]


def hamming_weight(values):
    values = np.asarray(values, dtype=np.uint8)
    flat = values.reshape(-1)
    weights = np.unpackbits(flat[:, np.newaxis], axis=1).sum(axis=1).astype(np.uint8)
    return weights.reshape(values.shape)


def _extract_byte_at_index(values, byte_index):
    values = np.asarray(values, dtype=np.uint8)
    if values.ndim == 0:
        return values
    if values.ndim == 1:
        return values
    if not 0 <= byte_index < values.shape[-1]:
        raise ValueError("byte_index is out of range for the provided byte array.")
    return np.take(values, byte_index, axis=-1)


def compute_aes_hw_pair(plaintext_bytes, key_bytes, byte_index=0):
    m_byte = _extract_byte_at_index(plaintext_bytes, byte_index)
    key_byte = _extract_byte_at_index(key_bytes, byte_index)
    y_values = aes_sbox(np.bitwise_xor(m_byte, key_byte))
    return hamming_weight(m_byte), hamming_weight(y_values)


def encode_joint_hw_labels(h_m, h_y, num_bits=8):
    base = num_bits + 1
    return np.asarray(h_m, dtype=np.int64) * base + np.asarray(h_y, dtype=np.int64)


def decode_joint_hw_labels(joint_labels, num_bits=8):
    base = num_bits + 1
    joint_labels = np.asarray(joint_labels, dtype=np.int64)
    return joint_labels // base, joint_labels % base


def evaluate_joint_predictions(predicted_joint_labels, reference_joint_labels, num_bits=8):
    predicted_joint_labels = np.asarray(predicted_joint_labels, dtype=np.int64)
    reference_joint_labels = np.asarray(reference_joint_labels, dtype=np.int64)
    predicted_h_m, predicted_h_y = decode_joint_hw_labels(predicted_joint_labels, num_bits=num_bits)
    reference_h_m, reference_h_y = decode_joint_hw_labels(reference_joint_labels, num_bits=num_bits)
    return {
        "accuracy_h_m": float(np.mean(predicted_h_m == reference_h_m)),
        "accuracy_h_y": float(np.mean(predicted_h_y == reference_h_y)),
        "joint_accuracy": float(np.mean(predicted_joint_labels == reference_joint_labels)),
    }


def print_metric_block(title, metrics):
    print(title)
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")


def train_validation_indices(labels, validation_fraction=0.2, random_state=0):
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    train_indices = []
    validation_indices = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        if class_indices.size >= 5:
            num_validation = max(1, int(round(class_indices.size * validation_fraction)))
        elif class_indices.size >= 2:
            num_validation = 1
        else:
            num_validation = 0
        validation_indices.extend(class_indices[:num_validation])
        train_indices.extend(class_indices[num_validation:])
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    if validation_indices.size == 0 and labels.size > 1:
        shuffled_indices = rng.permutation(labels.size)
        num_validation = max(1, int(round(labels.size * validation_fraction)))
        validation_indices = shuffled_indices[:num_validation]
        train_indices = shuffled_indices[num_validation:]
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def infer_single_attack_key_byte(key_matrix, byte_index=0):
    key_bytes = _extract_byte_at_index(key_matrix, byte_index).reshape(-1)
    unique_keys = np.unique(key_bytes)
    if unique_keys.size != 1:
        raise ValueError(f"Expected one fixed attack key byte, got {unique_keys.size} values.")
    return int(unique_keys[0])


def load_chipwhisperer_dataset(dataset_dir, invert_trace_polarity=True, max_rows=None):
    dataset_dir = Path(dataset_dir)
    traces = np.load(dataset_dir / "traces.npy")
    plaintexts = np.load(dataset_dir / "plain.npy")
    keys = np.load(dataset_dir / "key.npy")
    labels = np.load(dataset_dir / "labels.npy")
    if invert_trace_polarity:
        traces = -traces
    common_rows = min(traces.shape[0], plaintexts.shape[0], keys.shape[0], labels.shape[0])
    if max_rows is not None:
        common_rows = min(common_rows, int(max_rows))
    return {
        "traces": traces[:common_rows],
        "plaintexts": plaintexts[:common_rows],
        "keys": keys[:common_rows],
        "labels": labels[:common_rows],
        "inverted": bool(invert_trace_polarity),
    }


def display_notebook_figure(fig):
    try:
        from IPython.display import HTML, display
    except ImportError:
        if hasattr(fig, "show"):
            fig.show()
        return
    if hasattr(fig, "to_html"):
        display(HTML(fig.to_html(full_html=False, include_plotlyjs=True)))
    else:
        display(fig)

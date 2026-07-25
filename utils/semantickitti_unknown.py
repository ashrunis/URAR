import numpy as np
import torch
import yaml


UNKNOWN_SEMANTIC_SENTINEL = 66


def _load_semantickitti_yaml(label_mapping_path):
    with open(label_mapping_path, "r") as stream:
        return yaml.safe_load(stream)


def get_learning_label_name_map(label_mapping_path):
    semkittiyaml = _load_semantickitti_yaml(label_mapping_path)
    learning_map = semkittiyaml["learning_map"]
    raw_labels = semkittiyaml["labels"]

    learning_label_names = {}
    grouped_raw_ids = {}
    for raw_label, learning_label in sorted(learning_map.items()):
        if learning_label == 0:
            continue
        grouped_raw_ids.setdefault(learning_label, []).append(raw_label)

    for learning_label, raw_ids in grouped_raw_ids.items():
        learning_label_names[learning_label] = " / ".join(raw_labels[raw_id] for raw_id in raw_ids)

    return learning_label_names


def _normalize_unknown_labels(unknown_label=None, unknown_labels=None):
    if unknown_labels is not None and unknown_label is not None:
        raise ValueError("Use either unknown_label or unknown_labels, not both.")

    if unknown_labels is None:
        if unknown_label is None:
            unknown_labels = [5]
        else:
            unknown_labels = [unknown_label]

    normalized = [int(label) for label in unknown_labels]
    if len(normalized) == 0:
        raise ValueError("unknown_labels must contain at least one SemanticKITTI learning label.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"unknown_labels contains duplicates: {normalized}")
    return sorted(normalized)


def get_unknown_label_metadata(
    label_mapping_path,
    unknown_label=None,
    unknown_labels=None,
    num_known_classes=None,
):
    learning_label_names = get_learning_label_name_map(label_mapping_path)
    valid_learning_labels = sorted(learning_label_names.keys())
    unknown_labels = _normalize_unknown_labels(unknown_label, unknown_labels)

    invalid_unknown_labels = [label for label in unknown_labels if label not in learning_label_names]
    if invalid_unknown_labels:
        raise ValueError(
            f"Invalid SemanticKITTI unknown_labels={invalid_unknown_labels}. "
            f"Expected each label to be one of {valid_learning_labels}."
        )

    expected_known_classes = len(valid_learning_labels) - len(unknown_labels)
    if num_known_classes is not None and num_known_classes != expected_known_classes:
        raise ValueError(
            f"Configured num_class={num_known_classes}, but SemanticKITTI has "
            f"{expected_known_classes} known classes after removing unknown labels {unknown_labels}."
        )

    unique_label = np.asarray(valid_learning_labels) - 1
    unique_label_str = [learning_label_names[label] for label in valid_learning_labels]
    known_labels = [label for label in valid_learning_labels if label not in unknown_labels]
    known_label_indices = np.asarray(known_labels) - 1
    known_label_names = [learning_label_names[label] for label in known_labels]

    return {
        "unknown_label": unknown_labels[0],
        "unknown_label_name": learning_label_names[unknown_labels[0]],
        "unknown_labels": unknown_labels,
        "unknown_label_names": [learning_label_names[label] for label in unknown_labels],
        "unknown_labels_display": ", ".join(
            f"{label} ({learning_label_names[label]})" for label in unknown_labels
        ),
        "known_labels": known_labels,
        "known_label_indices": known_label_indices,
        "known_label_names": known_label_names,
        "num_known_classes": expected_known_classes,
        "unique_label": unique_label,
        "unique_label_str": unique_label_str,
    }


def collapse_unknown_label(labels, unknown_labels):
    collapsed = labels.clone()
    for unknown_label in unknown_labels:
        collapsed[collapsed == unknown_label] = 0
    for unknown_label in unknown_labels:
        collapsed = torch.where(collapsed > unknown_label, collapsed - 1, collapsed)
    return collapsed


def build_objectosphere_labels(labels, unknown_labels, unknown_label_sentinel=UNKNOWN_SEMANTIC_SENTINEL):
    obj_labels = labels.clone()
    obj_labels[obj_labels == 0] = -1
    for unknown_label in unknown_labels:
        obj_labels[obj_labels == unknown_label] = unknown_label_sentinel
    return obj_labels


def restore_known_predictions(prediction, known_labels):
    known_labels = np.asarray(known_labels, dtype=np.int64)
    return known_labels[prediction]

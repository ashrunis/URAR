import numpy as np
import torch
import yaml


def _indexed_values(mapping):
    keys = sorted(int(key) for key in mapping)
    if keys != list(range(keys[-1] + 1)):
        raise ValueError(f"nuScenes mapping keys must be contiguous from 0, got {keys}.")
    return [mapping[key] for key in keys]


def get_nuscenes_ood_metadata(label_mapping):
    with open(label_mapping, "r") as stream:
        config = yaml.safe_load(stream)

    inverse_values = np.asarray(
        _indexed_values(config["ood_inv_map"]),
        dtype=np.int64,
    )
    known_label_indices = inverse_values - 1
    label_names = config["labels_16"]
    return {
        "ood_map": config["ood_map"],
        "ood_obj_map": config["ood_obj_map"],
        "inverse_values": inverse_values,
        "known_labels": inverse_values,
        "known_label_indices": known_label_indices,
        "known_label_names": [label_names[int(label)] for label in inverse_values],
        "num_known_classes": len(inverse_values),
    }


def mapping_tensor(mapping, device):
    return torch.as_tensor(_indexed_values(mapping), dtype=torch.long, device=device)


def restore_known_predictions(predictions, inverse_values):
    return inverse_values[predictions]

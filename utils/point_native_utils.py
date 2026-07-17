import numpy as np
import torch
import torch.nn.functional as F

from utils.lovasz_losses import lovasz_softmax_flat
from utils.metric_util import fast_hist_crop
from utils.semantickitti_unknown import (
    build_objectosphere_labels,
    collapse_unknown_label,
    restore_known_predictions,
)


def pack_point_batch(point_coords, point_features, point_labels, device):
    """Concatenate a variable-size point-cloud batch without voxelizing it."""
    coords = []
    features = []
    labels = []
    batches = []
    lengths = []
    for batch_index, (sample_coords, sample_features, sample_labels) in enumerate(
        zip(point_coords, point_features, point_labels)
    ):
        sample_coords = torch.as_tensor(sample_coords, dtype=torch.float32, device=device)
        sample_features = torch.as_tensor(sample_features, dtype=torch.float32, device=device)
        sample_labels = torch.as_tensor(sample_labels, dtype=torch.long, device=device).reshape(-1)
        if sample_coords.ndim != 2 or sample_coords.shape[1] != 3:
            raise ValueError("Native point coordinates must have shape [num_points, 3].")
        if sample_features.ndim != 2 or sample_features.shape[0] != sample_coords.shape[0]:
            raise ValueError("Native point features must align with point coordinates.")
        if sample_labels.shape[0] != sample_coords.shape[0]:
            raise ValueError("Native point labels must align with point coordinates.")
        lengths.append(sample_coords.shape[0])
        coords.append(sample_coords)
        features.append(sample_features)
        labels.append(sample_labels)
        batches.append(
            torch.full(
                (sample_coords.shape[0],),
                batch_index,
                dtype=torch.long,
                device=device,
            )
        )

    return (
        torch.cat(coords, dim=0),
        torch.cat(features, dim=0),
        torch.cat(labels, dim=0),
        torch.cat(batches, dim=0),
        lengths,
    )


def urar_point_losses(
    css_logits,
    oss_logits,
    point_labels,
    unknown_labels,
    loss_sem_func,
    loss_objectosphere_func,
    loss_arcface_func,
):
    """Compute the URAR objectives on native point-model outputs."""
    css_labels = collapse_unknown_label(point_labels, unknown_labels)
    semantic_labels = css_labels - 1

    loss_sem = sum(loss_sem_func(css_logits, css_labels))
    valid = semantic_labels != -1
    loss_lovasz = lovasz_softmax_flat(
        F.softmax(css_logits[valid], dim=1),
        semantic_labels[valid],
    )
    loss_sem = loss_sem + loss_lovasz

    loss_arcface = loss_arcface_func(oss_logits, semantic_labels)
    objectosphere_labels = build_objectosphere_labels(point_labels, unknown_labels)
    loss_objectosphere = loss_objectosphere_func(oss_logits, objectosphere_labels)

    total = loss_sem + 0.3 * loss_objectosphere + 0.5 * loss_arcface
    return total, loss_sem, loss_objectosphere, loss_arcface


def known_class_histogram(css_logits, point_labels, point_lengths, known_labels):
    """Evaluate only known SemanticKITTI learning labels at original point order."""
    known_labels = np.asarray(known_labels, dtype=np.int64)
    known_indices = known_labels - 1
    predictions = restore_known_predictions(
        torch.argmax(css_logits, dim=1).detach().cpu().numpy(),
        known_labels,
    )
    labels = point_labels.detach().cpu().numpy()

    histogram = np.zeros((len(known_labels), len(known_labels)), dtype=np.int64)
    offset = 0
    for length in point_lengths:
        next_offset = offset + length
        histogram += fast_hist_crop(
            predictions[offset:next_offset],
            labels[offset:next_offset],
            known_indices,
        )
        offset = next_offset
    return histogram


def split_point_predictions(css_logits, oss_logits, point_lengths, known_labels):
    """Restore learning IDs and split raw-point predictions back per scan."""
    known_labels = np.asarray(known_labels, dtype=np.int64)
    css_predictions = restore_known_predictions(
        torch.argmax(css_logits, dim=1).detach().cpu().numpy(),
        known_labels,
    ).astype(np.int32)
    oss_logits = oss_logits.detach().cpu().numpy()

    result = []
    offset = 0
    for length in point_lengths:
        next_offset = offset + length
        result.append((css_predictions[offset:next_offset], oss_logits[offset:next_offset]))
        offset = next_offset
    return result

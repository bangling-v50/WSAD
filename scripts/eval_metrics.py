import numpy as np
from sklearn import metrics
from scipy import ndimage


def compute_imagewise_retrieval_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels):
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


def compute_aupro(
    anomaly_segmentations,
    ground_truth_masks,
    fpr_limit=0.3,
    num_thresholds=1000,
    connectivity=2,
):
    """Compute area under the per-region-overlap curve.

    The curve is limited to ``fpr_limit`` and normalized by the same limit.
    ``connectivity=2`` uses 8-neighbor connected components in two dimensions.
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    anomaly_segmentations = np.asarray(anomaly_segmentations, dtype=np.float32)
    ground_truth_masks = np.asarray(ground_truth_masks)

    # Allow [N, 1, H, W] or [N, H, W, 1].
    if anomaly_segmentations.ndim == 4 and anomaly_segmentations.shape[1] == 1:
        anomaly_segmentations = anomaly_segmentations[:, 0]
    if anomaly_segmentations.ndim == 4 and anomaly_segmentations.shape[-1] == 1:
        anomaly_segmentations = anomaly_segmentations[..., 0]

    if ground_truth_masks.ndim == 4 and ground_truth_masks.shape[1] == 1:
        ground_truth_masks = ground_truth_masks[:, 0]
    if ground_truth_masks.ndim == 4 and ground_truth_masks.shape[-1] == 1:
        ground_truth_masks = ground_truth_masks[..., 0]

    gt_masks = ground_truth_masks.astype(bool)
    scores = anomaly_segmentations.astype(np.float32)

    if scores.shape != gt_masks.shape:
        raise ValueError(
            f"Shape mismatch: anomaly_segmentations {scores.shape}, "
            f"ground_truth_masks {gt_masks.shape}"
        )

    if connectivity == 1:
        structure = np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            dtype=np.uint8,
        )
    else:
        structure = np.ones((3, 3), dtype=np.uint8)

    component_indices = []
    for img_idx in range(gt_masks.shape[0]):
        labeled_mask, num_components = ndimage.label(
            gt_masks[img_idx], structure=structure
        )
        for comp_id in range(1, num_components + 1):
            component = labeled_mask == comp_id
            if component.sum() > 0:
                component_indices.append(
                    (img_idx, np.flatnonzero(component.ravel()))
                )

    if not component_indices:
        raise ValueError(
            "No anomalous connected components found in ground_truth_masks."
        )

    normal_pixels = ~gt_masks
    num_normal_pixels = normal_pixels.sum()
    if num_normal_pixels == 0:
        raise ValueError("No normal pixels found. FPR cannot be computed.")

    score_min = float(scores.min())
    score_max = float(scores.max())
    thresholds = np.concatenate(
        [
            np.array([np.inf], dtype=np.float32),
            np.linspace(
                score_max,
                score_min,
                num_thresholds,
                dtype=np.float32,
            ),
        ]
    )

    fprs = []
    pros = []
    for threshold in thresholds:
        prediction = scores >= threshold
        false_positives = np.logical_and(prediction, normal_pixels).sum()
        fpr = false_positives / float(num_normal_pixels)

        pro_sum = 0.0
        for img_idx, component_flat_indices in component_indices:
            prediction_flat = prediction[img_idx].ravel()
            pro_sum += prediction_flat[component_flat_indices].mean()

        fprs.append(fpr)
        pros.append(pro_sum / len(component_indices))

    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]

    unique_fprs = []
    unique_pros = []
    for fpr in np.unique(fprs):
        repeated = fprs == fpr
        unique_fprs.append(fpr)
        unique_pros.append(pros[repeated].max())

    fprs = np.asarray(unique_fprs, dtype=np.float64)
    pros = np.asarray(unique_pros, dtype=np.float64)
    fpr_limit_actual = min(fpr_limit, fprs[-1])

    keep = fprs <= fpr_limit_actual
    fprs_limited = fprs[keep]
    pros_limited = pros[keep]

    if fprs_limited[-1] < fpr_limit_actual:
        pro_at_limit = np.interp(fpr_limit_actual, fprs, pros)
        fprs_limited = np.concatenate(
            [fprs_limited, [fpr_limit_actual]]
        )
        pros_limited = np.concatenate(
            [pros_limited, [pro_at_limit]]
        )

    if len(fprs_limited) < 2 or fpr_limit_actual == 0:
        aupro = 0.0
    else:
        aupro = (
            metrics.auc(fprs_limited, pros_limited) / fpr_limit_actual
        )

    return {
        "aupro": float(aupro),
        "fpr": fprs_limited,
        "pro": pros_limited,
        "fpr_limit": fpr_limit_actual,
    }

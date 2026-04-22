import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_loader import DatasetLoader, DatasetSplit
import eval_metrics as metrics


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def model_predict_placeholder(images: torch.Tensor):
    """
    Replace this function with your actual model inference.

    Input:
        images: Tensor of shape [B, 3, 224, 224]

    Output:
        image_scores: Tensor or ndarray of shape [B]
        pixel_maps:   Tensor or ndarray of shape [B, 224, 224]
    """
    batch_size = images.size(0)

    # ------------------------------------------------------------------
    # PSEUDOCODE:
    # 1. Feed images into your anomaly detection model
    # 2. Obtain one image-level anomaly score per image
    # 3. Obtain one pixel-level anomaly map per image
    # ------------------------------------------------------------------

    image_scores = torch.rand(batch_size)
    pixel_maps = torch.rand(batch_size, images.size(2), images.size(3))

    return image_scores, pixel_maps


def evaluate_wsad(data_root: str, batch_size: int = 8, num_workers: int = 0):
    """
    Evaluate WSAD with the released dataloader and metrics.

    Metrics:
        - Det:     image-level AUROC
        - Loc:     pixel-level AUROC on all test images
        - Loc_Ano: pixel-level AUROC on anomalous test images only
        - Avg:     macro average of class-wise pixel-level AUROC
    """

    # Build test dataset and dataloader.
    # Keep shuffle=False so that output order matches dataset order.
    test_dataset = DatasetLoader(
        source=data_root,
        split=DatasetSplit.TEST,
        imagesize=224,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    scores = []
    segmentations = []
    masks_gt = []
    anomaly_labels = []
    image_paths = []

    for batch in test_loader:
        images = batch["image"]          # [B, 3, 224, 224]
        masks = batch["mask"]            # [B, 1, 224, 224]
        labels = batch["is_anomaly"]     # [B]
        paths = batch["image_path"]

        # Replace with your actual inference.
        image_scores, pixel_maps = model_predict_placeholder(images)

        scores.extend(to_numpy(image_scores).tolist())
        segmentations.extend(to_numpy(pixel_maps))
        masks_gt.extend(to_numpy(masks).squeeze(1))
        anomaly_labels.extend(to_numpy(labels).astype(int).tolist())
        image_paths.extend(list(paths))

    # ------------------------------------------------------------
    # 1. Det: image-level AUROC
    # ------------------------------------------------------------
    det = metrics.compute_imagewise_retrieval_metrics(
        scores,
        anomaly_labels,
    )["auroc"]

    # ------------------------------------------------------------
    # 2. Loc: pixel-level AUROC on all test images
    # ------------------------------------------------------------
    loc = metrics.compute_pixelwise_retrieval_metrics(
        segmentations,
        masks_gt,
    )["auroc"]

    # ------------------------------------------------------------
    # 3. Loc_Ano: pixel-level AUROC on anomalous test images only
    # ------------------------------------------------------------
    sel_idxs = [i for i in range(len(masks_gt)) if np.sum(masks_gt[i]) > 0]

    loc_ano = metrics.compute_pixelwise_retrieval_metrics(
        [segmentations[i] for i in sel_idxs],
        [masks_gt[i] for i in sel_idxs],
    )["auroc"]

    # ------------------------------------------------------------
    # 4. Class-wise pixel-level AUROC
    #    Category name is inferred from the parent folder of each test image:
    #    e.g. data/test/CMC/xxx.jpg -> class "CMC"
    # ------------------------------------------------------------
    image_classes = [os.path.normpath(p).split(os.sep)[-2] for p in image_paths]

    classwise_segmentations = defaultdict(list)
    classwise_masks = defaultdict(list)

    for i in sel_idxs:
        cls = image_classes[i]
        classwise_segmentations[cls].append(segmentations[i])
        classwise_masks[cls].append(masks_gt[i])

    classwise_auroc = {}
    for cls in sorted(classwise_segmentations.keys()):
        classwise_auroc[cls] = metrics.compute_pixelwise_retrieval_metrics(
            classwise_segmentations[cls],
            classwise_masks[cls],
        )["auroc"]

    avg_classwise_auroc = float(np.mean(list(classwise_auroc.values())))

    results = {
        "Det": det,
        "Loc": loc,
        "Loc_Ano": loc_ano,
        "Avg": avg_classwise_auroc,
        "PerClass_Loc_AUROC": classwise_auroc,
    }

    return results


if __name__ == "__main__":
    # Example:
    # data_root should point to the directory containing:
    #   train/
    #   test/
    #   ground_truth/
    data_root = "data"

    results = evaluate_wsad(data_root=data_root, batch_size=8, num_workers=0)

    print(f"Det     : {results['Det']:.5f}")
    print(f"Loc     : {results['Loc']:.5f}")
    print(f"Loc_Ano : {results['Loc_Ano']:.5f}")
    print(f"Avg     : {results['Avg']:.5f}")

    print("Per-class Loc AUROC:")
    for cls, value in results["PerClass_Loc_AUROC"].items():
        print(f"  {cls}: {value:.5f}")
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
        - Det_I_AUROC: image-level AUROC
        - Loc_P_AUROC and Loc_AU_PRO: pixel metrics on all test images
        - Ano_P_AUROC and Ano_AU_PRO: pixel metrics on anomalous images
        - Macro_P_AUROC and Macro_AU_PRO: category macro averages

    The legacy keys Det, Loc, Loc_Ano, and Avg are retained as aliases for
    compatibility with the original conference release.
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

    # Pixel-level metrics on the complete test set (Loc.).
    loc_p_auroc = metrics.compute_pixelwise_retrieval_metrics(
        segmentations,
        masks_gt,
    )["auroc"]
    loc_au_pro = metrics.compute_aupro(
        segmentations,
        masks_gt,
    )["aupro"]

    # Pixel-level metrics on anomalous test images only (Ano.).
    sel_idxs = [i for i in range(len(masks_gt)) if np.sum(masks_gt[i]) > 0]
    ano_segmentations = [segmentations[i] for i in sel_idxs]
    ano_masks = [masks_gt[i] for i in sel_idxs]

    ano_p_auroc = metrics.compute_pixelwise_retrieval_metrics(
        ano_segmentations,
        ano_masks,
    )["auroc"]
    ano_au_pro = metrics.compute_aupro(
        ano_segmentations,
        ano_masks,
    )["aupro"]

    # Class-wise pixel-level AUROC and AU-PRO on anomalous images.
    #    Category name is inferred from the parent folder of each test image:
    #    e.g. data/test/CMC/xxx.jpg -> class "CMC"
    image_classes = [os.path.normpath(p).split(os.sep)[-2] for p in image_paths]

    classwise_segmentations = defaultdict(list)
    classwise_masks = defaultdict(list)

    for i in sel_idxs:
        cls = image_classes[i]
        classwise_segmentations[cls].append(segmentations[i])
        classwise_masks[cls].append(masks_gt[i])

    classwise_p_auroc = {}
    classwise_au_pro = {}
    for cls in sorted(classwise_segmentations.keys()):
        classwise_p_auroc[cls] = metrics.compute_pixelwise_retrieval_metrics(
            classwise_segmentations[cls],
            classwise_masks[cls],
        )["auroc"]
        classwise_au_pro[cls] = metrics.compute_aupro(
            classwise_segmentations[cls],
            classwise_masks[cls],
        )["aupro"]

    macro_p_auroc = float(np.mean(list(classwise_p_auroc.values())))
    macro_au_pro = float(np.mean(list(classwise_au_pro.values())))

    results = {
        "Det_I_AUROC": det,
        "Loc_P_AUROC": loc_p_auroc,
        "Loc_AU_PRO": loc_au_pro,
        "Ano_P_AUROC": ano_p_auroc,
        "Ano_AU_PRO": ano_au_pro,
        "Macro_P_AUROC": macro_p_auroc,
        "Macro_AU_PRO": macro_au_pro,
        "PerClass_Ano_P_AUROC": classwise_p_auroc,
        "PerClass_Ano_AU_PRO": classwise_au_pro,
        # Legacy aliases from the original conference release.
        "Det": det,
        "Loc": loc_p_auroc,
        "Loc_Ano": ano_p_auroc,
        "Avg": macro_p_auroc,
        "PerClass_Loc_AUROC": classwise_p_auroc,
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

    print(f"Det. I-AUROC : {results['Det_I_AUROC']:.5f}")
    print(f"Loc. P-AUROC : {results['Loc_P_AUROC']:.5f}")
    print(f"Loc. AU-PRO  : {results['Loc_AU_PRO']:.5f}")
    print(f"Ano. P-AUROC : {results['Ano_P_AUROC']:.5f}")
    print(f"Ano. AU-PRO  : {results['Ano_AU_PRO']:.5f}")
    print(f"Macro P-AUROC: {results['Macro_P_AUROC']:.5f}")
    print(f"Macro AU-PRO : {results['Macro_AU_PRO']:.5f}")

    print("Per-class metrics:")
    for cls in results["PerClass_Ano_P_AUROC"]:
        p_auroc = results["PerClass_Ano_P_AUROC"][cls]
        au_pro = results["PerClass_Ano_AU_PRO"][cls]
        print(f"  {cls}: P-AUROC={p_auroc:.5f}, AU-PRO={au_pro:.5f}")

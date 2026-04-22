# WSAD: Wood Surface Anomaly Detection Dataset

WSAD is a wood surface anomaly detection dataset for unsupervised evaluation. 
It is constructed for wood surface defect detection and localization under the one-class setting.

The released version contains cropped image patches rather than raw production-line images. 
It includes normal training images, normal test images, anomalous test images, and pixel-level masks for anomalous test samples.

## Dataset Summary

- Training set: 1000 normal images
- Test set: 120 normal images and 120 anomalous images
- Pixel-level masks are provided for anomalous test images only
- Normal test images do not have explicit mask files
- The dataset contains 10 defect categories:
  - CMC: coarse burr
  - CP: delamination
  - GB: dry scar
  - LF: crack
  - LX: internal decay
  - QK: chip
  - QMC: shallow burr
  - SJ: knot
  - SMC: deep burr
  - SP: bark inclusion

WSAD contains 1,120 normal images and 120 anomalous images across 10 defect categories with pixel-level ground-truth masks, and the official protocol uses 1,000 normal images for training while evaluating on the remaining normal images together with all anomalous images.

## Directory Structure

```text
WSAD/
├── README.md
├── LICENSE
├── CITATION.cff
└── data/
    ├── train/
    │   └── good/
    ├── test/
    │   ├── good/
    │   ├── CMC/
    │   ├── CP/
    │   ├── GB/
    │   ├── LF/
    │   ├── LX/
    │   ├── QK/
    │   ├── QMC/
    │   ├── SJ/
    │   ├── SMC/
    │   └── SP/
    └── ground_truth/
        ├── CMC/
        ├── CP/
        ├── GB/
        ├── LF/
        ├── LX/
        ├── QK/
        ├── QMC/
        ├── SJ/
        ├── SMC/
        └── SP/
```

## Annotation

Under the guidance of an expert with many years of experience in manual wood defect inspection and sorting, the anomalous region in each defect image was first roughly outlined by hand to determine its approximate extent. GrabCut was then applied to refine the foreground defect region, and the segmentation result was manually checked and corrected when necessary. The released masks are binary annotations, where foreground pixels indicate anomalous regions and background pixels indicate normal regions.

For defects with clear boundaries, the masks follow the visible defect contours as closely as possible. 
For low-contrast or blurred-boundary defects, the masks mainly cover the visually identifiable anomalous region.

Each anomalous test image has a corresponding mask file. 
Normal test images do not have explicit mask files and can be treated as all-zero masks during pixel-level evaluation.

## Official Protocol

WSAD is designed for unsupervised anomaly detection under the one-class setting.

- Only normal images are used for training
- Test images include both normal and anomalous samples
- Pixel-level evaluation is conducted using the provided ground-truth masks
- Normal test images can be treated as all-zero masks during pixel-level evaluation
- All input images are resized to 224 × 224
- Ground-truth masks are resized accordingly using nearest-neighbor interpolation

## Data Organization

You can load the dataset according to the directory structure above.

- `data/train/good/` contains normal training images
- `data/test/good/` contains normal test images
- `data/test/<defect_type>/` contains anomalous test images
- `data/ground_truth/<defect_type>/` contains the corresponding pixel-level masks

## Evaluation Example

WSAD can be evaluated by combining the released dataloader with image-level anomaly scores and pixel-level anomaly maps predicted by a model.

- `Det`: image-level AUROC
- `Loc`: pixel-level AUROC on all test images
- `Loc_Ano`: pixel-level AUROC on anomalous test images only
- `Avg`: macro average of class-wise pixel-level AUROC

A minimal evaluation example is provided in `scripts/eval_example.py`.

## Notes

- The released dataset contains cropped image patches rather than raw production-line images.
- Only anomalous test images have explicit mask files.
- Normal test images are regarded as negative samples in image-level evaluation and can be treated as all-zero masks in pixel-level evaluation.
- Users should follow the official protocol above for fair comparison.

## License

This dataset is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

See the `LICENSE` file for details.

## Citation

If you use this dataset in research, please cite the associated paper and this repository.

Citation information is provided in `CITATION.cff`.
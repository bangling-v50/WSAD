# WSAD: Wood Surface Anomaly Detection Dataset

WSAD is a wood surface anomaly detection dataset for unsupervised evaluation.
It was constructed from images of rotary-cut eucalyptus veneers acquired on the production line of a single wood-processing plant and is intended for wood surface defect detection and localization under the one-class setting.

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
- Each defect category contains 12 anomalous images

WSAD contains 1,120 normal images and 120 anomalous images across 10 defect categories with pixel-level ground-truth masks, and the official protocol uses 1,000 normal images for training while evaluating on the remaining normal images together with all anomalous images.

## Data Acquisition and Dataset Construction

WSAD was constructed from rotary-cut eucalyptus veneers supplied by the same wood-processing plant. Each veneer measured 1270 mm × 640 mm × 2.2 mm. The source images were acquired on a production line at a resolution of 4096 × 3000 pixels under a fixed camera arrangement, light-shielding conditions, and area-light illumination.

All normal images in the released dataset have a resolution of 224 × 224 pixels. They were obtained from the source images using a 224 × 224-pixel sliding window with horizontal and vertical strides of 196 pixels. The last windows on the right and bottom were aligned with the corresponding image boundaries. Anomalous images were cropped from the source production-line images and manually screened. To preserve complete defect regions, some anomalous images use larger crop sizes. Most anomalous images have a resolution of 224 × 224 pixels, while a small number have resolutions of 256 × 256, 512 × 512, or 1041 × 1041 pixels. Samples that could not be clearly assigned to a defect category were excluded.

The training and test sets were formed from the same batch of eucalyptus veneers, using source images acquired on different dates, and contain no duplicate or spatially overlapping crops. The public release contains only the cropped image patches and their available annotations. The original high-resolution production-line images are not publicly released because they contain commercially sensitive information and are subject to confidentiality obligations with the industrial partner.

## Directory Structure

```text
WSAD/
├── README.md
├── CHANGELOG.md
├── LICENSE
├── CITATION.cff
├── scripts/
│   ├── dataset_loader.py
│   ├── eval_metrics.py
│   └── main.py
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

Under the guidance of an expert in manual wood defect inspection and sorting, the anomalous region in each defect image was first roughly outlined by hand to determine its approximate extent. GrabCut was then applied to refine the foreground defect region, and the segmentation result was manually checked and corrected when necessary. Each released mask has the same dimensions as its corresponding image and is stored as a binary annotation, with a pixel value of 255 for anomalous regions and 0 for the background.

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

## File Naming and Image-Mask Matching

- Normal training images follow `data/train/good/<stem>.jpg`.
- Normal test images follow `data/test/good/<stem>.jpg` and do not have explicit mask files.
- Anomalous test images follow `data/test/<defect_type>/<stem>.jpg`.
- Their masks follow `data/ground_truth/<defect_type>/<stem>_mask.png`.
- Image stems may repeat across defect categories. Each anomalous image must therefore be matched to a mask within the corresponding defect-type directory.

## Evaluation Example

WSAD can be evaluated by combining the released dataloader with image-level anomaly scores and pixel-level anomaly maps predicted by a model.

The extended evaluation reports the following metrics:

- `Det_I_AUROC`: image-level AUROC over the complete test set
- `Loc_P_AUROC`: pixel-level AUROC over the complete test set
- `Loc_AU_PRO`: AU-PRO over the complete test set
- `Ano_P_AUROC`: pixel-level AUROC on anomalous test images only
- `Ano_AU_PRO`: AU-PRO on anomalous test images only
- `Macro_P_AUROC`: macro average of category-level pixel AUROC
- `Macro_AU_PRO`: macro average of category-level AU-PRO

AU-PRO is normalized over the false-positive-rate interval from 0 to 0.3. Ground-truth regions use 8-neighbor connectivity, and the curve is evaluated using 1,000 uniformly spaced score thresholds.

For compatibility with the original conference release, the existing output keys remain available: `Det` is an alias of `Det_I_AUROC`, `Loc` of `Loc_P_AUROC`, `Loc_Ano` of `Ano_P_AUROC`, and `Avg` of `Macro_P_AUROC`.

A minimal evaluation example is provided in `scripts/main.py`. The example includes `model_predict_placeholder`, which generates random scores only to demonstrate the expected prediction interface. It must be replaced with actual model inference before the evaluation results are meaningful.

See `CHANGELOG.md` for changes to the released documentation and evaluation example.

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

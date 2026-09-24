# Changelog

## 2026-09-24

### Dataset documentation

- Added acquisition and construction details for the released WSAD image patches.
- Clarified image sizes, train-test separation, mask values, and image-mask matching.
- Clarified that the original high-resolution production-line images are not publicly released because of industrial confidentiality requirements.

### Evaluation example

- Added AU-PRO for the complete test set and anomalous test images only.
- Added category-level pixel AUROC and AU-PRO together with their macro averages.
- Documented the AU-PRO settings: an FPR limit of 0.3, 1,000 thresholds, and 8-neighbor connectivity.
- Retained the original `Det`, `Loc`, `Loc_Ano`, `Avg`, and `PerClass_Loc_AUROC` result keys as backward-compatible aliases.

These additions extend the public evaluation example used with the journal manuscript. They do not alter the released images, masks, directory structure, or the original conference results.

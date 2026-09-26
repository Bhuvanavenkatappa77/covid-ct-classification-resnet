# COVID-19 Detection from Lung CT Scans (PyTorch, ResNet-50)

Binary classification of lung CT images (COVID-19 vs. non-COVID) with a ResNet-50 CNN, comparing **training from scratch** against **transfer learning** from ImageNet weights, with **Grad-CAM** to see which regions the model looks at.

## Results

Validation set: 60 CT images (balanced).

| Approach | Validation accuracy | Sensitivity (non-COVID / COVID) | Precision (non-COVID / COVID) |
|---|---|---|---|
| ResNet-50 trained from scratch | **83.3%** | 1.00 / 0.67 | 0.75 / 1.00 |
| ResNet-50 transfer learning (ImageNet) | **96.7%** | 0.93 / 1.00 | 1.00 / 0.94 |

Transfer learning improved accuracy by more than 13 points, and recall on COVID cases went from 67% to 100%. With only ~2,000 training images, pretrained features generalize much better than learning everything from zero.

## Approach

**Data:** 2,482 lung CT images resized to 224×224. Split: 2,022 train / 60 validation / 400 test.

**Model:** torchvision ResNet-50 with the final layer replaced by a single-logit output (`Linear(2048, 1)`) for binary classification.

| | From scratch | Transfer learning |
|---|---|---|
| Initial weights | Random | ImageNet (`ResNet50_Weights.DEFAULT`) |
| Trainable layers | All | `layer4` + final `fc` only (rest frozen) |
| Loss | Binary cross-entropy with logits | Same |
| Optimizer | SGD, lr 1e-4, momentum 0.99 | Same |

- Checkpoint saved every epoch; best epoch selected by validation accuracy.
- Evaluation with a confusion matrix, per-class sensitivity and precision.

**Explainability:** Grad-CAM and EigenCAM on the last ResNet block (`layer4`) produce heatmaps over the CT scan, to check that predictions are based on lung regions and not image artifacts.

## Files

| File | Contents |
|---|---|
| `test(CNN from scratch).ipynb` | ResNet-50 from random init, training loop, evaluation, Grad-CAM |
| `test1(Transfer learning).ipynb` | Pretrained ResNet-50, partial fine-tuning, evaluation, Grad-CAM / EigenCAM |

## How to run

```bash
pip install torch torchvision scikit-image matplotlib pandas grad-cam
jupyter notebook
```

The CT images are not included in this repo. Point the dataset paths in the notebooks to your local copy of the images.

## Tech

Python · PyTorch · torchvision · ResNet-50 · Transfer learning · Grad-CAM · NumPy · pandas · scikit-image · Matplotlib

## What I learned

- Transfer learning is the right default when labeled medical images are scarce.
- Freezing early layers and fine-tuning only the last block trains faster and overfits less.
- Accuracy alone hides errors: the scratch model missed a third of COVID cases, which only the per-class sensitivity showed.
- Explainability tools like Grad-CAM matter in healthcare, where you need to see why a model decides.

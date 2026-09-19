# BioFusion-ScanSSM-Explainable-CNN-Transformer-SSM-for-Cervical-Cytology-Classification
BioFusion-ScanSSM for Explainable Cervical Cell Classification
# BioFusion-ScanSSM

## Explainable CNN–Transformer–State Space Framework for Cervical Cytology Classification

BioFusion-ScanSSM is a hybrid deep-learning framework for automated cervical-cell classification from Pap smear images. The proposed architecture combines hierarchical convolutional feature extraction, global semantic representation learning, gated cross-attention fusion, and a biologically conditioned state-space model for efficient long-range dependency modelling.

The framework also incorporates cytology-aware preprocessing, morphology-consistent feature learning, spatial knowledge distillation, and post-hoc explainability using Grad-CAM++, SHAP, and LIME.

> This repository is intended for research and educational use. It is not a clinically validated diagnostic system and must not be used as a substitute for professional cytological assessment.

---

## Key Features

- Hybrid CNN–Transformer–State Space architecture.
- Dynamic Patch-Aware Gamma Correction for illumination and contrast adaptation.
- Macenko stain normalization for reducing staining variation.
- Cytoplasm masking to emphasize diagnostically relevant cellular structures.
- Gated Cross-Attention Fusion for adaptive local–global feature integration.
- Linear Bio-SSSM for efficient long-range spatial dependency modelling.
- Nucleus Feature Consistency loss for morphology-aware feature geometry.
- Spatial Feature Distillation for transferring predictive and spatial knowledge.
- Compact dual-DenseNet student model for deployment-oriented inference.
- Four-view test-time augmentation.
- Explainability using Grad-CAM++, SHAP, and LIME.
- Evaluation on the SIPaKMeD cervical cytology dataset.

---

## Framework Overview

BioFusion-ScanSSM follows a teacher–student learning paradigm.

The full BioFusion-ScanSSM model acts as the teacher and combines:

1. Cytology-aware image preprocessing.
2. Hierarchical CNN-based local morphology extraction.
3. Global semantic feature modelling.
4. Gated Cross-Attention Fusion.
5. A six-layer Linear Bio-SSSM tower.
6. Teacher-guided spatial and predictive knowledge transfer.

A compact dual-DenseNet model serves as the student. After distillation, the student model can be used independently for inference without requiring the full teacher architecture.

```text
Pap smear image
      │
      ▼
Cytology-aware preprocessing
(DPAGC, stain normalization, cytoplasm masking)
      │
      ├──────────────────────┐
      ▼                      ▼
Hierarchical CNN       Global feature backbone
local features         semantic features
      │                      │
      └──────────┬───────────┘
                 ▼
      Gated Cross-Attention Fusion
                 │
                 ▼
          Linear Bio-SSSM tower
                 │
                 ▼
       BioFusion-ScanSSM teacher
                 │
        Spatial Feature Distillation
                 │
                 ▼
       Dual-DenseNet student model
                 │
                 ▼
        Five-class prediction
```

---

## Dataset

The primary experiments use the publicly available SIPaKMeD dataset.

The dataset contains 4,049 isolated cervical-cell images distributed across five classes:

| Class | Description |
|---|---|
| `Dyskeratotic` | Abnormal dyskeratotic cervical cells |
| `Koilocytotic` | Abnormal koilocytotic cervical cells |
| `Metaplastic` | Benign metaplastic cervical cells |
| `Parabasal` | Normal parabasal cervical cells |
| `Superficial-Intermediate` | Normal superficial intermediate cervical cells |

The experiments use a stratified 90:10 train–test split with random seed `42`.

### Dataset preparation

After downloading the dataset, organize the images using the following structure:

```text
data/
└── SIPaKMeD/
    ├── Dyskeratotic/
    ├── Koilocytotic/
    ├── Metaplastic/
    ├── Parabasal/
    └── Superficial-Intermediate/
```

The exact directory names may be changed in the configuration file if they differ from the original dataset organization.

---

## Preprocessing

The image-processing pipeline contains the following stages:

1. Dynamic Patch-Aware Gamma Correction.
2. Macenko stain normalization.
3. Cytoplasm masking.
4. Resizing and center cropping.
5. Tensor conversion.
6. Training-set channel normalization.
7. Conditional Positional Encoding where applicable.

The default model input size is:

```text
224 × 224 × 3
```

The training pipeline may additionally apply random geometric augmentation, including horizontal flipping and small-angle rotations.

---

## Model Components

### Gated Cross-Attention Fusion

G-CAF uses local CNN features as queries and global semantic features as keys and values. A learnable sigmoid gate controls the relative contribution of the global context and local morphological representation.

The fused representation is given by:

\[
F_{\text{fused}}
=
g \odot F_{\text{attn}}
+
(1-g)\odot F_{\text{CNN}},
\]

where \(g\) is a learnable token-level gate.

This enables the model to retain local nuclear morphology while selectively incorporating relevant global context.

### Linear Bio-SSSM

The Linear Bio-SSSM module models long-range token dependencies using a parallel associative scan formulation. The module uses morphology-conditioned state updates to modulate the contribution of local features based on global cytological information.

The module is designed to provide efficient spatial dependency modelling while avoiding the quadratic complexity of full self-attention.

### Nucleus Feature Consistency Loss

The NFC loss encourages:

- Intra-class compactness.
- Inter-class separation.
- Morphology-aware organization of the latent feature space.

The objective is combined with the classification and distillation losses during training.

### Spatial Feature Distillation

The teacher transfers two forms of information to the student:

- Temperature-scaled class probabilities.
- Normalized spatial feature representations.

The combined distillation objective is:

\[
\mathcal{L}_{\text{SFD}}
=
\frac{1}{2}\mathcal{L}_{\text{KD}}
+
\frac{1}{2}\mathcal{L}_{\text{spatial}}.
\]

The teacher remains frozen during student optimization.

---

## Installation

Clone the repository:

```bash
git clone [https://github.com/](https://github.com/)<your-username>/biofusion-scanssm.git
cd biofusion-scanssm
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

A representative environment includes:

```text
Python 3.9+
PyTorch 2.x
TorchVision
NumPy
scikit-learn
Pillow
OpenCV
tqdm
PyYAML
matplotlib
seaborn
shap
lime
```

For GPU training, install a PyTorch version compatible with the CUDA version available on your system.

---

## Repository Structure

```text
biofusion-scanssm/
├── configs/
│   ├── default.yaml
│   └── sipakmed.yaml
├── data/
│   └── README.md
├── datasets/
│   ├── sipakmed_dataset.py
│   └── preprocessing.py
├── models/
│   ├── biofusion_scanssm.py
│   ├── bio_sssm.py
│   ├── gated_cross_attention.py
│   ├── dual_densenet.py
│   └── positional_encoding.py
├── losses/
│   ├── nfc_loss.py
│   ├── focal_loss.py
│   └── distillation_loss.py
├── explainability/
│   ├── gradcam_pp.py
│   ├── shap_explanations.py
│   └── lime_explanations.py
├── train_teacher.py
├── train_student.py
├── eval_student.py
├── evaluate_tta.py
├── requirements.txt
├── LICENSE
└── README.md
```

The structure above is a recommended organization and should be synchronized with the actual repository files.

---

## Training

### Train the BioFusion-ScanSSM teacher

```bash
python train_teacher.py \
    --config configs/sipakmed.yaml \
    --data-root data/SIPaKMeD \
    --output-dir checkpoints/teacher
```

### Train the dual-DenseNet student

```bash
python train_student.py \
    --config configs/sipakmed.yaml \
    --data-root data/SIPaKMeD \
    --teacher-checkpoint checkpoints/teacher/biofusion_scanssm_best.pth \
    --output-dir checkpoints/student
```

During student training, the teacher is frozen and the student is optimized using classification, focal, NFC, and spatial feature distillation losses.

---

## Evaluation

Evaluate the student model using standard single-view inference:

```bash
python eval_student.py \
    --data-root data/SIPaKMeD \
    --checkpoint checkpoints/student/sipakmed_student_dual_densenet_best.pth \
    --split test
```

Evaluate using four-view test-time augmentation:

```bash
python evaluate_tta.py \
    --data-root data/SIPaKMeD \
    --checkpoint checkpoints/student/sipakmed_student_dual_densenet_best.pth \
    --views original hflip rotate_plus rotate_minus
```

The four-view TTA configuration uses:

- Original image.
- Horizontally flipped image.
- Image rotated by \(+10^\circ\).
- Image rotated by \(-10^\circ\).

Predictions are obtained by averaging the probability vectors from all views.

---

## Reported Results

The reported results are obtained on the fixed stratified SIPaKMeD test split.

| Metric | Standard evaluation | Four-view TTA |
|---|---:|---:|
| Accuracy | 97.53% | 97.78% |
| Macro-F1 | 0.9755 | 0.9779 |
| Weighted-F1 | 0.9753 | 0.9777 |
| Cohen’s kappa | 0.9691 | 0.9722 |
| Macro-AUC | 0.9982 | 0.9990 |

The most challenging category was Abnormal Koilocytotic, which showed greater morphological overlap with benign metaplastic and normal superficial-intermediate cells.

These values should be reproduced using the released code, preprocessing settings, checkpoint-selection protocol, and fixed data split before being used for scientific comparison.

---

## Explainability

The repository supports three post-hoc explanation methods:

### Grad-CAM++

Grad-CAM++ visualizations highlight image regions that contribute to a selected class prediction.

```bash
python explainability/gradcam_pp.py \
    --checkpoint checkpoints/student/sipakmed_student_dual_densenet_best.pth \
    --input path/to/image.png \
    --output-dir outputs/gradcam
```

### SHAP

SHAP can be used to estimate feature contributions for individual predictions or groups of test images.

```bash
python explainability/shap_explanations.py \
    --checkpoint checkpoints/student/sipakmed_student_dual_densenet_best.pth \
    --input path/to/image.png \
    --output-dir outputs/shap
```

### LIME

LIME generates local superpixel-based explanations for individual predictions.

```bash
python explainability/lime_explanations.py \
    --checkpoint checkpoints/student/sipakmed_student_dual_densenet_best.pth \
    --input path/to/image.png \
    --output-dir outputs/lime
```

The explanations are intended for model-inspection and research purposes. Visual agreement between explanation methods does not establish clinical validity or explanation faithfulness.

---

## Reproducibility

For reproducible experiments:

- Use random seed `42`.
- Use the same stratified train–test split.
- Calculate normalization statistics using only the training subset.
- Maintain the input resolution of `224 × 224`.
- Use the same checkpoint-selection criterion.
- Record the PyTorch, CUDA, cuDNN, and GPU versions.
- Report both standard inference and TTA results separately.
- Avoid data leakage between training, validation, and test images.
- Preserve patient-level separation when patient metadata is available.

---

## Limitations

The current implementation and evaluation have several limitations:

- The primary dataset contains isolated cervical-cell images rather than complete whole-slide images.
- The model does not currently perform cell detection, segmentation, or slide-level aggregation.
- External cervical-domain performance may be affected by staining, scanner, population, and class-distribution shifts.
- The Herlev result is lower than the SIPaKMeD result, indicating challenges in cross-dataset cervical cytology generalization.
- The model requires additional independent, multi-center validation before clinical use.
- The explanation results are primarily qualitative and have not been validated by a cytopathologist reader study.
- The reported performance represents cell-level classification and should not be interpreted as patient-level screening performance.

---

## Citation

If you use this repository or the BioFusion-ScanSSM framework in your research, please cite the associated manuscript:

```bibtex
@article{singh2026biofusion,
  title   = {BioFusion-ScanSSM: An Intelligence-Driven Hybrid CNN-Transformer-State Space Framework with Gated Cross-Attention Fusion for Explainable Cervical Cancer Classification from Pap Smear Images},
  author  = {Singh, Davinder Paul and Banerjee, Tathagat and Reddy, Santosh P. and Kumar, Senthil J. and Pasupulla, Ajay Prakash},
  year    = {2026},
  note    = {Manuscript under review}
}
```

Please replace the citation metadata with the final journal, conference, DOI, and repository information after publication.

---

## Ethical and Clinical Disclaimer

This software is a research prototype for automated cervical-cell image classification. It has not been approved for clinical diagnosis, patient management, or screening decisions.

The predictions generated by this software must not be used independently of qualified healthcare professionals. Any clinical deployment requires independent external validation, regulatory assessment, prospective evaluation, calibration analysis, and review by qualified cytopathologists.

---

## License

Add the intended open-source license before publishing the repository.

For example:

```text
MIT License
```

If the dataset or pretrained models have additional licensing restrictions, those terms must also be followed.

---

## Acknowledgements

The experiments use the SIPaKMeD cervical cytology dataset. The authors acknowledge the dataset creators and the institutions that contributed to its development and public availability.

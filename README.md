# Synthetic Data-Assisted Road Damage Detection for Indian Roads

A machine learning project investigating whether adding synthetic road-damage images to real training data can improve YOLO-based road-damage detection on unseen real-world Indian road images.

## Project Overview

Road damage such as longitudinal cracks, transverse cracks, alligator cracks, and potholes can affect road safety and increase vehicle maintenance costs. Manual inspection is time-consuming, so this project explores automated road-damage detection using computer vision.

We compare two YOLO11n models:

- **Model A — Real-only:** trained using real road-damage images.
- **Model B — Real + synthetic:** trained using real images supplemented with additional data from a computer-vision dataset and LLM-assisted synthetic image generation.

Both models are evaluated on the same fixed test set of real images to make the comparison consistent.

## Research Question

**Does adding synthetic road-damage data improve detection performance on unseen real-world road images?**

### Research Hypotheses

- **Null hypothesis (\(H_0\)):** Adding synthetic road-damage images to the real training dataset does not improve detection performance on unseen real-world road images.
- **Alternative hypothesis (\(H_1\)):** Adding synthetic road-damage images to the real training dataset improves detection performance on unseen real-world road images.

## Road-Damage Classes

The project detects four classes:

| Class ID | Class |
|---:|---|
| 0 | Longitudinal crack |
| 1 | Transverse crack |
| 2 | Alligator crack |
| 3 | Pothole |

## Dataset Sources and Preparation

### Real dataset

The project uses the Indian portion of the Road Damage Dataset 2022 (RDD2022).

- [RDD2022 dataset on Figshare](https://doi.org/10.6084/m9.figshare.21431547.v1)
- [RDD2022 research paper](https://doi.org/10.1002/gdj3.260)
- [Official RoadDamageDetector repository](https://github.com/sekilab/RoadDamageDetector)

The real dataset was split into training, validation, and test sets. The same real validation and test sets are retained for the model comparison.

### Additional computer-vision and LLM-assisted data

The additional data is organized into two sources:

- `Computer_Vision_Dataset` — downloaded computer-vision data.
- `LLM_Dataset` — images generated with the assistance of LLM-created prompts.

These are stored under `synthetic_data_for _YOLO`. Dataset preparation includes checking image–label filename correspondence. The matching script uses the final numeric identifier in filenames and copies candidate matching labels to `train_labels_matched` folders, leaving original labels unchanged. Numeric matching should be verified before training.

**Note:** Record the exact public URL and image-generation model used for the additional datasets here if they are part of the final submission.

## Methodology

1. **Collect data:** Obtain real Indian road images and prepare additional computer-vision and LLM-assisted images.
2. **Preprocess:** Organize images and YOLO bounding-box annotations, and inspect image–label matches.
3. **Train the baseline:** Fine-tune a pretrained YOLO11n model using real data only.
4. **Train the augmented model:** Fine-tune YOLO11n using real training images plus synthetic/additional training images.
5. **Evaluate fairly:** Evaluate both models on the same fixed real-world test set.
6. **Compare:** Review precision, recall, mAP@50, and mAP@50–95.

Synthetic/additional images are used for training only; the original real validation and test sets are not augmented with synthetic images.

## Experimental Setup

The models were trained using Ultralytics YOLO11n with the following configuration:

| Setting | Value |
|---|---|
| Model | YOLO11n |
| Epochs | 30 |
| Image size | 512 |
| Batch size | 16 |
| Device | Apple Silicon MPS |
| Workers | 2 |
| Random seed | 42 |
| AMP | Disabled |

The real-only baseline was selected from the completed `real_only_fast_mps-2` run. The real-plus-synthetic model was trained in the `real_plus_synthetic_mps` run.

## Test Results

The following results were reported from evaluation on the same fixed real test set:

| Metric | Model A: Real-only | Model B: Real + synthetic | Difference |
|---|---:|---:|---:|
| Precision | 0.2931 | 0.5980 | +0.3050 |
| Recall | 0.3292 | 0.3297 | +0.0006 |
| mAP@50 | 0.3008 | 0.3134 | +0.0126 |
| mAP@50–95 | 0.1275 | 0.1341 | +0.0066 |

### Initial interpretation

- Precision increased substantially for the real-plus-synthetic model.
- Recall remained almost unchanged.
- mAP@50 and mAP@50–95 improved modestly.
- These results are preliminary evidence, not proof of a statistically significant improvement. Further analysis, such as per-class results and repeated runs or paired uncertainty estimates, is needed before making a strong statistical claim.

## Repository Structure

A suggested repository layout is:

```text
road-damage-detection/
├── README.md
├── notebooks/
│   └── road_damage_detection.ipynb
├── configs/
│   ├── data.yaml
│   └── data_real_synthetic.yaml
├── scripts/
│   ├── prepare_dataset.py
│   ├── match_image_labels.py
│   ├── train_real_only.py
│   ├── train_real_synthetic.py
│   └── evaluate_models.py
├── results/
│   ├── model_comparison.csv
│   └── figures/
└── requirements.txt
```

This is a suggested structure; update it to match the files actually committed to the repository.

## Environment and Installation

Python and the following packages are used in the project:

- `ultralytics`
- `torch`
- `opencv-python`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `PyYAML`

Create and activate a virtual environment, then install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install ultralytics opencv-python numpy pandas matplotlib scikit-learn pyyaml
```

Install a PyTorch build compatible with your operating system and hardware by following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Reproducing the Experiments

1. Download and prepare the real dataset and annotations.
2. Prepare the additional computer-vision and LLM-assisted data.
3. Check the YAML configuration paths and class names.
4. Train Model A on the real training split.
5. Train Model B on the real training split plus the additional training images.
6. Evaluate both trained checkpoints against the same real test split.
7. Compare the metrics and inspect the per-class evaluation outputs.

Example Ultralytics training pattern:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(
    data="path/to/data.yaml",
    epochs=30,
    imgsz=512,
    batch=16,
    device="mps",  # Apple Silicon; use an appropriate device on other hardware
    workers=2,
    seed=42,
    amp=False,
)
```

Update dataset paths for your own machine. Do not commit absolute local paths that only work on one computer.

## Limitations

- The results depend on dataset quality, class distribution, and annotation accuracy.
- Numeric filename matching assumes that matching identifiers represent the same image; this must be checked.
- Synthetic images can contain unrealistic visual patterns or incorrect labels.
- A single training run does not quantify run-to-run variability.
- Performance on a held-out test set does not guarantee reliability across all Indian roads, weather, or lighting conditions.

## Future Work

- Compare per-class precision, recall, and mAP.
- Inspect false positives and missed detections.
- Repeat training with multiple random seeds and quantify uncertainty.
- Verify synthetic-image quality and annotation correctness.
- Evaluate on additional real-world road images from different locations and conditions.

## Acknowledgements

The project builds on the publicly available RDD2022 road-damage dataset and the Ultralytics YOLO framework. Please retain the original dataset and software citations and comply with their respective licences and terms of use.

## License

Add a repository license after confirming the licences and usage terms of all datasets, generated assets, and code included in this project.

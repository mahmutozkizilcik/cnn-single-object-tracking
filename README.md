# CNN-Based Single-Object Tracking (SOT) Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Array%20Processing-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=for-the-badge)](https://matplotlib.org/)

An end-to-end **Deep Learning & Computer Vision** project implementing a **Single-Object Tracking (SOT)** engine on video sequences. Unlike general object detection where all instances of a category are detected across frames, this system tracks a specific target object initialized solely by a bounding box in the first frame across subsequent video frames using **CNN-based spatial feature extraction** and **bounding-box regression**.

---

## 🎯 System Architecture & Methodology

The tracking pipeline combines localized search regions with deep regression networks to accurately estimate target transformations over time:

```
[Frame t-1 State] ──> [Search Region Extraction] ──> [CNN Feature Extractor] ──> [Bounding-Box Regressor] ──> [Frame t Prediction]
                              ▲                                                                                      │
                              └──────────────────────── Target Center & Scale Update ────────────────────────────────┘
```

### Key Components

1. **Sequence & Data Processing**:
   - Parses video frame sequences (`img/*.jpg`) alongside ground-truth target coordinates (`groundtruth_rect.txt`).
   - Normalizes bounding box definitions $[x, y, w, h]$ and converts between corner and center-scale coordinates.

2. **Search-Region Sampling**:
   - Extracts localized search windows around the predicted target location from frame $t-1$ to track the object in frame $t$.
   - Applies padding and scale adjustments to provide context while maintaining computational efficiency.

3. **CNN Bounding-Box Regression**:
   - Employs a Convolutional Neural Network (CNN) architecture trained to predict coordinate residuals $(\Delta x, \Delta y, \Delta w, \Delta h)$ relative to the search window.
   - Learns spatial features robust to target appearance changes and motion dynamics.

4. **Temporal Tracking Loop**:
   - Iterates frame-by-frame across validation and test sequences without requiring online fine-tuning.
   - Dynamically updates search window centers based on previous regression predictions.

---

## 📊 Performance & Visual Analysis

The tracker's robustness is evaluated across diverse visual challenges, including fast motion, scale variations, background clutter, and temporary occlusions.

| Evaluation Metric | Description | Key Observations |
| :--- | :--- | :--- |
| **Center Location Error (CLE)** | Euclidean distance between predicted bounding box center and ground truth center. | Remains sub-pixel accurate during smooth translational motion; minor spikes during sudden direction changes. |
| **Intersection over Union (IoU)** | Spatial overlap ratio between predicted bounding box and ground-truth bounding box. | High overlap consistency (>0.75 average) across standard sequences; demonstrates resilience against mild deformation. |
| **Drift Resistance** | Ability to maintain track without permanent loss of target over prolonged sequences. | Stable tracking achieved by localized search sampling, preventing background drift in cluttered scenes. |

### Visual Examples

Sample visual tracking outputs and dataset visualizations are stored in `outputs/` and `pa2 submit/videos and photos/`:
* **Search Region Samples (`search_region_sample.png`)**: Demonstration of cropped context regions used as CNN inputs.
* **Tracking Visualizations (`good_results.png`, `dataset_viz_car2.png`)**: Overlay comparisons of predicted bounding boxes vs. ground-truth trajectories across video frames.

---

## 📁 Repository Structure

```text
├── student_draft.ipynb              # Main implementation notebook containing data loading, CNN training & evaluation
├── pa2 submit/
│   ├── mahmut-özkızılcık-pa2.ipynb  # Final submitted tracking pipeline & experiment runs
│   └── videos and photos/           # Visual evaluation outputs, plots, and qualitative results
├── outputs/                         # Generated plots and visualization assets
├── .gitignore                       # Excludes heavy datasets (>1.6GB), PyTorch checkpoints (.pth), and copyrighted PDFs
└── README.md                        # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.10+ installed along with standard deep learning packages:

```bash
pip install torch torchvision numpy opencv-python matplotlib jupyter
```

### Running the Tracking Pipeline

1. **Open Jupyter Notebook**:
   Launch the workspace and open either `student_draft.ipynb` or `pa2 submit/mahmut-özkızılcık-pa2.ipynb`.
   ```bash
   jupyter notebook
   ```

2. **Dataset Setup**:
   Place your tracking sequence directories (containing `img/` frames and `groundtruth_rect.txt`) inside the project root or configure the dataset path in the notebook setup cell. *(Note: Dataset files and pre-trained weights `.pth` are excluded from Git via `.gitignore` to comply with GitHub size limits and copyright policies).*

3. **Execute Tracking Cells**:
   Run the notebook sequentially to train the regression model and visualize tracking performance on test sequences.

---

## ⚖️ Note on Copyright & Repository Integrity

This repository contains purely original code implementations, experiment logs, and visual analysis. Institutional assignment specification sheets (`*.pdf`), proprietary dataset archives (`*.zip`), and heavy model checkpoint binaries (`*.pth`) are strictly excluded via `.gitignore` to maintain academic integrity and repository hygiene.

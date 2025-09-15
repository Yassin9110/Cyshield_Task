# Stereo and Monocular Depth Estimation Evaluation

This project provides scripts and tools to evaluate monocular and stereo depth estimation models on a structured dataset with ground truth depth and disparity maps.

## Project Structure

```
.
├── evaluate_monocular.py
├── evaluate_stereo.py
├── requirements.txt
├── common/
│   ├── metrics.py
│   └── utils.py
├── data/
│   ├── data_description.md
│   ├── sensor_config.json
│   └── images/
├── monocular/
│   └── models.py
├── stereo/
│   ├── raft_stereo.py
│   └── stereo_models.py
├── Monocular_evaluation_results/
└── Stereo_evaluation_results/
```

## Dataset Format

- **Images:**  
  - `data/images/left/` and `data/images/right/`: Stereo image pairs.
  - `data/images/gt_depth/`: Ground truth depth maps (in meters).
  - `data/images/gt_disp/`: Ground truth disparity maps (in pixels).

- **Camera Calibration:**  
  - `data/sensor_config.json`: Contains intrinsic matrix `K`, baseline, and image dimensions. See [data/data_description.md](data/data_description.md) for details.

## Installation

1. Clone this repository.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Monocular Depth Evaluation

Run:
```sh
python evaluate_monocular.py
```
- Evaluates all models listed in `MODELS_TO_EVALUATE` on the left images.
- Results and visualizations are saved in `Monocular_evaluation_results/`.

### Stereo Depth Evaluation

Run:
```sh
python evaluate_stereo.py
```
- Evaluates stereo models (e.g., SGBM, RAFT-Stereo) on stereo pairs.
- Results and visualizations are saved in `Stereo_evaluation_results/`.

## Output

- **CSV Reports:**  
  - `evaluation_results_detailed.csv`: Per-image metrics.
  - `evaluation_results_summary.csv`: Per-model average metrics.

- **Visualizations:**  
  - Predicted depth/disparity maps and error maps for each image and model.
  - Ground truth visualizations.

## Adding New Models

- **Monocular:**  
  Add model loading and inference logic in [`monocular/models.py`](monocular/models.py).

- **Stereo:**  
  Add new predictors in [`stereo/stereo_models.py`](stereo/stereo_models.py).


---

For more information on dataset structure and configuration, see [data/data_description.md](data/data_description.md).
# evaluate_stereo.py

import os
import glob
import time
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Import from our new modular structure
from common.utils import load_image, save_visualized_map
from common.metrics import compute_metrics
from stereo.stereo_models import RAFTStereoPredictor, predict_sgbm, load_sensor_config, disparity_to_depth

# ------------------- USER CONFIG -------------------
DATASET_PATH = "data/"
OUTPUT_PATH = "/content/Stereo_evaluation_results/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_DISP_CLAMP = 0.1 # To avoid division by zero

# Path to the pre-trained RAFT-Stereo model
RAFT_STEREO_CKPT = 'stereo/raftstereo-middlebury.pth'

# SGBM Parameters
SGBM_CONFIG = {
    'minDisparity': 1, 'numDisparities': 128, 'blockSize': 5,
    'P1': 8 * 1 * 5**2, 'P2': 32 * 1 * 5**2, # Using 1 channel for grayscale
    'disp12MaxDiff': 1, 'uniquenessRatio': 10, 'speckleWindowSize': 100,
    'speckleRange': 32, 'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
}
# ----------------------------------------------------

def create_stereo_validity_mask(gt_depth_raw, gt_disp):
    """Creates a validity mask specific to the stereo GT conventions."""
    invalid_codes = (gt_depth_raw == 0) | (gt_depth_raw == 255)
    valid_gt_mask = (gt_disp > 0)
    mask = (~invalid_codes) & valid_gt_mask
    return mask

def main():
    print("--- Running Stereo Matching Evaluation ---")
    print(f"Using device: {DEVICE}")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "ground_truth_visualized"), exist_ok=True)

    image_ids = sorted([Path(f).stem for f in glob.glob(os.path.join(DATASET_PATH, "images/left/*.png"))])
    if not image_ids:
        print(f"No images found in {os.path.join(DATASET_PATH, 'images/left/')}. Exiting.")
        return

    # --- Initialize Models ---
    try:
        raft_predictor = RAFTStereoPredictor(RAFT_STEREO_CKPT, DEVICE)
        models_to_evaluate = {"SGBM": None, "RAFT-Stereo": raft_predictor}
    except (ImportError, FileNotFoundError) as e:
        print(f"Warning: Could not initialize RAFT-Stereo predictor ({e}). Evaluating SGBM only.")
        models_to_evaluate = {"SGBM": None}

    # --- Main Loop ---
    all_results = []
    for image_id in tqdm(image_ids, desc="Processing Images"):
        left_path = os.path.join(DATASET_PATH, "images/left", f"{image_id}.png")
        right_path = os.path.join(DATASET_PATH, "images/right", f"{image_id}.png")
        gt_depth_path = os.path.join(DATASET_PATH, "images/gt_depth", f"{image_id}.png")
        gt_disp_path = os.path.join(DATASET_PATH, "images/gt_disp", f"{image_id}.png")
        calib_path = os.path.join(DATASET_PATH, "sensor_config.json")

        left_img_gray = load_image(left_path, cv2.IMREAD_GRAYSCALE)
        right_img_gray = load_image(right_path, cv2.IMREAD_GRAYSCALE)
        gt_depth_raw = load_image(gt_depth_path, cv2.IMREAD_UNCHANGED)
        gt_disp_raw = load_image(gt_disp_path, cv2.IMREAD_UNCHANGED)
        fx, baseline = load_sensor_config(calib_path)

        gt_depth = gt_depth_raw.astype(np.float32)
        gt_disp = gt_disp_raw.astype(np.float32)

        validity_mask = create_stereo_validity_mask(gt_depth_raw, gt_disp)
        if not validity_mask.any():
            print(f"Warning: no valid GT pixels for {image_id}. Skipping.")
            continue

        for model_name, predictor in models_to_evaluate.items():
            model_out_dir = os.path.join(OUTPUT_PATH, model_name)
            os.makedirs(model_out_dir, exist_ok=True)

            start_time = time.time()
            if model_name == "SGBM":
                pred_disp = predict_sgbm(left_img_gray, right_img_gray, SGBM_CONFIG)
            elif model_name == "RAFT-Stereo":
                pred_disp = predictor.predict(left_path, right_path)
            inference_time = time.time() - start_time
            
            pred_depth = disparity_to_depth(pred_disp, fx, baseline, MIN_DISP_CLAMP)

            metrics = compute_metrics(
                pred_depth=pred_depth, gt_depth=gt_depth, mask=validity_mask,
                pred_disp=pred_disp, gt_disp=gt_disp  # Pass disparity maps for EPE
            )

            row = {"image_id": image_id, "model_name": model_name, "inference_time": inference_time, **metrics}
            all_results.append(row)

            # --- Save visualizations ---
            save_visualized_map(pred_disp, os.path.join(model_out_dir, f"{image_id}_pred_disp.png"))
            save_visualized_map(pred_depth, os.path.join(model_out_dir, f"{image_id}_pred_depth.png"))

    # --- Save GT visualizations (only once) ---
    for image_id in image_ids[:1]: # Just do for one image to show it works
        gt_depth_path = os.path.join(DATASET_PATH, "images/gt_depth", f"{image_id}.png")
        gt_disp_path = os.path.join(DATASET_PATH, "images/gt_disp", f"{image_id}.png")
        gt_depth = load_image(gt_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        gt_disp = load_image(gt_disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        gt_out_dir = os.path.join(OUTPUT_PATH, "ground_truth_visualized")
        save_visualized_map(gt_depth, os.path.join(gt_out_dir, f"{image_id}_gt_depth.png"))
        save_visualized_map(gt_disp, os.path.join(gt_out_dir, f"{image_id}_gt_disp.png"))

    # --- Generate and Save CSV Reports ---
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_PATH, "evaluation_results_detailed.csv")
    results_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nDetailed results saved to: {csv_path}")

    summary_df = results_df.drop(columns=['image_id']).groupby('model_name').mean()
    print("\n--- Overall Stereo Model Performance (Averages) ---")
    print(summary_df.to_string(float_format='%.4f'))
    
    summary_csv_path = os.path.join(OUTPUT_PATH, "evaluation_results_summary.csv")
    summary_df.to_csv(summary_csv_path, float_format='%.4f')
    print(f"Summary saved to: {summary_csv_path}")
    print("Stereo evaluation complete.")

if __name__ == "__main__":
    main()
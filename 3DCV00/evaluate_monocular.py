# evaluate_monocular.py

import os
import glob
import time
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import from our new modular structure
from common.utils import load_image, save_visualized_map
from common.metrics import compute_metrics
from monocular.models import get_model_and_processor, run_inference, align_and_scale

# ------------------- USER CONFIG -------------------
DATASET_PATH = "data/"
OUTPUT_PATH = "Monocular_depth_evaluation_results/"
MODELS_TO_EVALUATE = ["MiDaS_small", "ZoeDepth_N", "DPT_Hybrid"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GT_MIN_DEPTH = 2.0
GT_MAX_DEPTH = 164.0
# ----------------------------------------------------

def create_mono_validity_mask(gt_depth_raw, gt_depth):
    """Creates a validity mask specific to the monocular GT conventions."""
    invalid_codes = (gt_depth_raw == 0) | (gt_depth_raw == 255)
    mask = (~invalid_codes) & (gt_depth >= GT_MIN_DEPTH) & (gt_depth < GT_MAX_DEPTH)
    return mask

def main():
    print(f"--- Running Monocular Depth Evaluation ---")
    print(f"Using device: {DEVICE}")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    for m in MODELS_TO_EVALUATE:
        os.makedirs(os.path.join(OUTPUT_PATH, m), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "ground_truth_visualized"), exist_ok=True)

    left_images = sorted(glob.glob(os.path.join(DATASET_PATH, "images/left/*.png")))
    gt_depths_paths = sorted(glob.glob(os.path.join(DATASET_PATH, "images/gt_depth/*.png")))
    gt_depth_map = {os.path.basename(p): p for p in gt_depths_paths}

    if not left_images:
        print(f"No images found in {os.path.join(DATASET_PATH, 'images/left/')}. Exiting.")
        return

    all_results = []
    for model_name in MODELS_TO_EVALUATE:
        model, processor = get_model_and_processor(model_name, DEVICE)
        model.eval()

        for img_path in tqdm(left_images, desc=f"Evaluating {model_name}"):
            img_name = os.path.basename(img_path)
            if img_name not in gt_depth_map:
                print(f"Warning: missing GT depth for {img_name}. Skipping.")
                continue

            rgb_image = cv2.cvtColor(load_image(img_path), cv2.COLOR_BGR2RGB)
            gt_depth_raw = load_image(gt_depth_map[img_name], cv2.IMREAD_UNCHANGED)
            gt_depth = gt_depth_raw.astype(np.float32)

            validity_mask = create_mono_validity_mask(gt_depth_raw, gt_depth)
            if not validity_mask.any():
                print(f"Warning: no valid GT pixels for {img_name}. Skipping.")
                continue

            start_time = time.time()
            pred_raw = run_inference(model, processor, rgb_image, model_name, DEVICE)
            inference_time = time.time() - start_time

            # Post-processing: ZoeDepth is metric, others need alignment
            if model_name == "ZoeDepth_N":
                pred_depth_metric = pred_raw
            else:
                pred_depth_metric = align_and_scale(pred_raw, gt_depth, validity_mask)

            pred_depth_metric = np.clip(pred_depth_metric, GT_MIN_DEPTH, GT_MAX_DEPTH)

            metrics = compute_metrics(pred_depth_metric, gt_depth, validity_mask)
            
            row = {"image_name": img_name, "model_name": model_name, "inference_time": inference_time, **metrics}
            all_results.append(row)

            # --- Save visualizations ---
            model_out_dir = os.path.join(OUTPUT_PATH, model_name)
            img_stem = os.path.splitext(img_name)[0]

            save_visualized_map(
                pred_depth_metric,
                os.path.join(model_out_dir, f"{img_stem}_pred_depth.png"),
                vmax=GT_MAX_DEPTH
            )

            error_map = np.abs(pred_depth_metric - gt_depth)
            error_map[~validity_mask] = np.nan # Use NaN for invalid areas for visualization
            save_visualized_map(
                error_map,
                os.path.join(model_out_dir, f"{img_stem}_error_map.png")
            )
            
            # Save GT visualization once per image, during the first model's run
            if model_name == MODELS_TO_EVALUATE[0]:
                save_visualized_map(
                    gt_depth,
                    os.path.join(OUTPUT_PATH, "ground_truth_visualized", f"{img_stem}_gt_depth.png"),
                    vmax=GT_MAX_DEPTH
                )

    # --- Generate and Save CSV Reports ---
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_PATH, "evaluation_results_detailed.csv")
    results_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nDetailed results saved to: {csv_path}")

    summary_df = results_df.drop(columns=['image_name']).groupby('model_name').mean()
    print("\n--- Overall Monocular Model Performance (Averages) ---")
    print(summary_df.to_string(float_format='%.4f'))
    
    summary_csv_path = os.path.join(OUTPUT_PATH, "evaluation_results_summary.csv")
    summary_df.to_csv(summary_csv_path, float_format='%.4f')
    print(f"Summary saved to: {summary_csv_path}")
    print("Monocular evaluation complete.")

if __name__ == "__main__":
    main()
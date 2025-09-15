# common/metrics.py

import numpy as np

def compute_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray, pred_disp: np.ndarray = None, gt_disp: np.ndarray = None):
    """
    Compute depth evaluation metrics and optionally End-Point-Error for disparity.

    Args:
        pred_depth: Predicted depth map.
        gt_depth: Ground truth depth map.
        mask: Boolean validity mask.
        pred_disp: (Optional) Predicted disparity map for EPE calculation.
        gt_disp: (Optional) Ground truth disparity map for EPE calculation.
    """
    # Ensure inputs are float for calculations
    pred_d = pred_depth[mask].astype(np.float32)
    gt_d = gt_depth[mask].astype(np.float32)

    # --- Standard Depth Metrics (from monocular script) ---
    abs_rel = float(np.mean(np.abs(pred_d - gt_d) / gt_d))
    rmse = float(np.sqrt(np.mean((pred_d - gt_d) ** 2)))
    mae = float(np.mean(np.abs(pred_d - gt_d)))
    rmse_log = float(np.sqrt(np.mean((np.log(pred_d + 1e-8) - np.log(gt_d + 1e-8)) ** 2)))
    log_diff = np.log(pred_d + 1e-8) - np.log(gt_d + 1e-8)
    silog = float(np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2)))
    ratio = np.maximum(pred_d / gt_d, gt_d / pred_d)
    delta1 = float((ratio < 1.25).mean())
    delta2 = float((ratio < 1.25 ** 2).mean())
    delta3 = float((ratio < 1.25 ** 3).mean())

    metrics = {
        "AbsRel": abs_rel,
        "RMSE": rmse,
        "MAE": mae,
        "RMSElog": rmse_log,
        "SILog": silog,
        "δ<1.25": delta1,
        "δ<1.25²": delta2,
        "δ<1.25³": delta3,
    }

    # --- Disparity Metric (EPE), if inputs are provided ---
    if pred_disp is not None and gt_disp is not None:
        pred_disp_valid = pred_disp[mask].astype(np.float32)
        gt_disp_valid = gt_disp[mask].astype(np.float32)
        
        if gt_disp_valid.size > 0:
            epe = float(np.mean(np.abs(pred_disp_valid - gt_disp_valid)))
            metrics["EPE"] = epe
        else:
            metrics["EPE"] = np.nan

    return metrics
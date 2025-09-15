# monocular/model_handler.py

import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, DPTImageProcessor, DPTForDepthEstimation

def get_model_and_processor(model_name: str, device: str):
    """Loads a monocular depth estimation model and its preprocessor."""
    print(f"\nLoading model: {model_name}...")
    if model_name == "MiDaS_small":
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device)
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        processor = transforms.small_transform
        return model, processor
    elif model_name == "ZoeDepth_N":
        processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
        model = AutoModelForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to(device)
        return model, processor
    elif model_name == "DPT_Hybrid":
        processor = DPTImageProcessor.from_pretrained("intel/dpt-hybrid-midas")
        model = DPTForDepthEstimation.from_pretrained("intel/dpt-hybrid-midas").to(device)
        return model, processor
    else:
        raise ValueError(f"Model '{model_name}' not supported.")

def run_inference(model, processor, image: np.ndarray, model_name: str, device: str):
    """Runs inference for a given monocular model."""
    h, w, _ = image.shape
    model.eval()
    with torch.no_grad():
        if model_name == "MiDaS_small":
            inp = processor(image).to(device)
            pred = model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            ).squeeze()
            return pred.cpu().numpy()
        elif model_name in ["DPT_Hybrid", "ZoeDepth_N"]:
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
            pred = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            ).squeeze()
            return pred.cpu().numpy()
        else:
            raise ValueError(f"Inference for '{model_name}' not supported.")

def align_and_scale(pred: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray):
    """
    Aligns a relative depth prediction to the ground truth using median scaling.
    This is for models like MiDaS and DPT that output inverse, scale-less depth.
    """
    pred_inv_depth = 1.0 / (pred + 1e-8)
    scale = np.median(gt_depth[mask]) / np.median(pred_inv_depth[mask])
    pred_metric = scale * pred_inv_depth
    return pred_metric

# stereo/model_handler.py

import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from .raft_stereo import RAFTStereo
print("Imported RAFT-Stereo successfully.")
from .raft_stereo import InputPadder


class RAFTStereoPredictor:
    def __init__(self, ckpt_path, device):
        # A simple namespace object to mimic the original args
        from types import SimpleNamespace
        args = SimpleNamespace(
            restore_ckpt=ckpt_path, mixed_precision=False, valid_iters=32,
            hidden_dims=[128]*3, corr_implementation='alt', shared_backbone=False,
            corr_levels=4, corr_radius=4, n_downsample=2, context_norm='batch',
            slow_fast_gru=False, n_gru_layers=3
        )
        self.args = args
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        # NOTE: This is a placeholder for the actual RAFT-Stereo import
        # You need to have the model definition available.
        # This will fail if the imports at the top of the file fail.
        print("Loading RAFT-Stereo model...")

        model = nn.DataParallel(RAFTStereo(self.args))
        if not os.path.exists(self.args.restore_ckpt):
            raise FileNotFoundError(f"RAFT-Stereo checkpoint not found at {self.args.restore_ckpt}")
        model.load_state_dict(torch.load(self.args.restore_ckpt, map_location=torch.device(self.device)))
        model = model.module
        model.to(self.device)
        model.eval()
        print("RAFT-Stereo model loaded successfully.")
        return model

    def predict(self, left_img_path, right_img_path):


        def load_image_for_raft(imfile):
            img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            return img[None].to(self.device)

        with torch.no_grad():
            image1 = load_image_for_raft(left_img_path)
            image2 = load_image_for_raft(right_img_path)
            padder = InputPadder(image1.shape, divis_by=32)
            image1_padded, image2_padded = padder.pad(image1, image2)
            _, flow_up = self.model(image1_padded, image2_padded, iters=self.args.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()
            disparity = -flow_up.cpu().numpy()
        return disparity

def predict_sgbm(left_img, right_img, sgbm_config):
    """Computes disparity using OpenCV's SGBM algorithm."""
    sgbm = cv2.StereoSGBM_create(**sgbm_config)
    pred_disp = sgbm.compute(left_img, right_img).astype(np.float32) / 16.0
    return pred_disp

def load_sensor_config(path):
    """Loads camera intrinsics and baseline from a JSON file."""
    with open(path, "r") as f:
        cfg = json.load(f)
    fx = cfg["K"][0][0]
    baseline = cfg["baseline"]
    return fx, baseline

def disparity_to_depth(disp, fx, baseline, min_disp_clamp=0.1):
    """Converts a disparity map to a depth map."""
    disp = np.maximum(disp, min_disp_clamp)
    depth = (fx * baseline) / disp
    return depth
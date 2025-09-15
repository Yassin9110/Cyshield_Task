# common/utils.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

def load_image(path: str, flags=cv2.IMREAD_UNCHANGED):
    """Loads an image from a path."""
    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def save_visualized_map(data: np.ndarray, path: str, colormap="inferno", vmin=None, vmax=None):
    """
    Save a 2D float map with a colormap. Maps to [0,1] for stability.
    """
    if np.all(np.isnan(data)):
        # If all data is NaN, save a black image
        h, w = data.shape
        blank = np.zeros((h, w), dtype=np.uint8)
        plt.imsave(path, blank, cmap='gray')
        return

    # Use percentile for robust auto-scaling if vmin/vmax are not provided
    if vmin is None:
        vmin = np.nanpercentile(data, 2)
    if vmax is None:
        vmax = np.nanpercentile(data, 98)

    # Normalize data to [0, 1] range
    norm = (data - vmin) / (vmax - vmin + 1e-8)
    norm = np.clip(norm, 0.0, 1.0)

    # Convert to 8-bit unsigned byte format for saving
    arr8 = img_as_ubyte(norm)
    plt.imsave(path, arr8, cmap=colormap)

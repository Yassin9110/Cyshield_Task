# ───────────────────────────── imports ──────────────────────────────────
import os
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn       
import torchvision.transforms as transforms
from VAE import Encoder, Decoder

# ───────────────────── device / model loading ───────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

latent_dim = 512

# Your Encoder / Decoder classes must already be defined somewhere above
encoder = Encoder(latent_dim).to(device)
decoder = Decoder(latent_dim).to(device)

checkpoint = torch.load("model.pt", map_location=device)
encoder.load_state_dict(checkpoint["encoder_states"])
decoder.load_state_dict(checkpoint["decoder_states"])
encoder.eval()
decoder.eval()


# ───────────────────────── helper functions ─────────────────────────────
transform = transforms.Compose([
    transforms.Resize((178, 218)),  # adapt to the VAE's input size
    transforms.ToTensor(),
])


def load_and_preprocess_image(img_source) -> torch.Tensor:
    """Accepts a path or PIL.Image, returns tensor (1,C,H,W) on the right device."""
    if isinstance(img_source, (str, Path)):
        img = Image.open(img_source).convert("RGB")
    else:                               # assume already a PIL image
        img = img_source
    return transform(img).unsqueeze(0).to(device)


@torch.no_grad()
def encode_image(x: torch.Tensor) -> torch.Tensor:
    mu, sigma = encoder(x)
    std = torch.exp(0.5 * sigma)
    eps = torch.randn_like(std)
    return mu + eps * std               # re-parameterisation trick


@torch.no_grad()
def decode_latent(z: torch.Tensor) -> np.ndarray:
    img = decoder(z)
    img = torch.squeeze(img).permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0.0, 1.0)


def generate_variations(z0: torch.Tensor,
                        k: int,
                        strength: float) -> List[torch.Tensor]:
    return [z0 + torch.randn_like(z0) * strength for _ in range(k)]


def save_image(img_np: np.ndarray, fname: Path | str):
    Image.fromarray((img_np * 255).astype(np.uint8)).save(fname)


def numpy_back_to_tensor(img_np: np.ndarray) -> torch.Tensor:
    pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
    return load_and_preprocess_image(pil_img)


# ───────────────────────── core generation ──────────────────────────────
def generate_images(image_path: str | Path,
                    num_images: int = 10,
                    mode: str = "variation",
                    variation_strength: float = 0.3,
                    out_dir: str | Path = "results") -> List[np.ndarray]:
    """
    Returns a list with all saved images (numpy format in [0,1]).
    In 'variation' mode image_00_reconstruction.jpg is always the first element.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs_out: list[np.ndarray] = []

    if mode == "variation":
        x = load_and_preprocess_image(image_path)
        z0 = encode_image(x)

        recon = decode_latent(z0)
        save_image(recon, out_dir / "image_00_reconstruction.jpg")
        imgs_out.append(recon)

        for i, z in enumerate(generate_variations(z0, num_images, variation_strength), 1):
            img = decode_latent(z)
            save_image(img, out_dir / f"image_{i:02d}.jpg")
            imgs_out.append(img)

    elif mode == "cycle":
        x = load_and_preprocess_image(image_path)
        for i in range(1, num_images + 1):
            z = encode_image(x)
            img = decode_latent(z)
            save_image(img, out_dir / f"image_{i:02d}.jpg")
            imgs_out.append(img)
            x = numpy_back_to_tensor(img)

    else:
        raise ValueError("mode must be 'variation' or 'cycle'")

    return imgs_out


# ──────────────────────────── MAIN API ──────────────────────────────────
def main(image_path: str, mode: str = "variation", num_images: int = 10, variation_strength: float = 0.3, out_dir: str = "results",
         make_mosaic: bool = True, mosaic_cols: int = 4, mosaic_file: str = "mosaic.jpg") -> List[np.ndarray]:
    """
    Generate images with a pre-trained VAE in two modes:
    1) 'variation': add Gaussian noise in latent space around the reference image
    2) 'cycle': encode → decode → encode … for N steps

    Parameters:
    image_path: Path to the reference image
    mode: Either 'variation' or 'cycle'
    num_images: Number of images to generate
    variation_strength: Noise strength (for variation mode only)
    out_dir: Output directory for generated images
    make_mosaic: Whether to create and save a mosaic of all generated images
    mosaic_cols: Number of columns in the mosaic
    mosaic_file: Filename for the mosaic image

    Returns:
    List of generated images as numpy arrays
    """
    images = generate_images(
        image_path=image_path,
        num_images=num_images,
        mode=mode,
        variation_strength=variation_strength,
        out_dir=out_dir,
    )

    if make_mosaic:
        rows = (len(images) + mosaic_cols - 1) // mosaic_cols
        fig, ax = plt.subplots(rows, mosaic_cols,
                               figsize=(mosaic_cols * 4, rows * 4))
        ax = ax.ravel()
        for i, img in enumerate(images):
            ax[i].imshow(img)
            ax[i].set_title(f"{i:02d}")
            ax[i].axis("off")
        for i in range(len(images), len(ax)):
            ax[i].axis("off")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / mosaic_file)
        plt.show()

    return images

if __name__ == "__main__":
    images = main(
        image_path="ronaldo.jpg",
        mode="cycle",
        num_images=10,
        variation_strength=0.2,
        out_dir="cycle_results"
    )
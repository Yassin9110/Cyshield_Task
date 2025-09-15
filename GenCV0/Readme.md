# Variational Autoencoder Image Generator

This project uses a pre-trained Variational Autoencoder (VAE) to generate new images based on a reference image. It supports two modes:
- **Variation**: Generates variations of an input image by adding Gaussian noise in the latent space.
- **Cycle**: Repeatedly encodes and decodes an image, showing how the image evolves over multiple cycles.

## Project Structure

```
main.py
VAE.py
model.pt
ronaldo.jpg
cycle_results/
variation_results/
...
```

- `main.py`: Main script for image generation.
- `VAE.py`: Contains the `Encoder` and `Decoder` class definitions.
- `model.pt`: Pre-trained VAE model weights.
- `ronaldo.jpg`: Example input image.
- `cycle_results/`, `variation_results/`: Output directories for generated images.

## Usage

1. **Install dependencies**  
   ```
   pip install -r requirements.txt
   ```

2. **Run the script**  
   ```
   python main.py
   ```
   By default, this will generate 10 cycled images from `ronaldo.jpg` and save them in `cycle_results/`.

3. **Custom usage**  
   You can modify the parameters in the `main()` call at the bottom of `main.py`:
   - `image_path`: Path to the input image.
   - `mode`: `"variation"` or `"cycle"`.
   - `num_images`: Number of images to generate.
   - `variation_strength`: Noise strength (for variation mode).
   - `out_dir`: Output directory.

## Output

- Individual generated images are saved as `.jpg` files in the specified output directory.
- A mosaic of all generated images is also saved as `mosaic.jpg` in the output directory.

## Requirements

See [requirements.txt](requirements.txt) for dependencies.

## Notes

- The VAE model architecture must match the one used to train `model.pt`.
- Make sure `VAE.py` defines the `Encoder` and `Decoder` classes with the correct `latent_dim`.

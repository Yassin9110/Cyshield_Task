# Age-Invariant Face Recognition (AIFR) & Age Prediction Web App

This project provides a web application for **face recognition** and **age prediction** using deep learning models. The app allows users to upload two facial images, predicts the age for each, and verifies if both images belong to the same person, even across age gaps.

## Features

- **Face Detection & Alignment:** Uses MTCNN for robust face detection and alignment.
- **Age Prediction:** Predicts the age of each detected face using a ResNet-based regression model.
- **Face Verification:** Compares two faces using an ArcFace-based recognition model to determine if they are the same person.
- **Web Interface:** Built with Streamlit for easy interaction.

## Project Structure

```
aifr-age-predictor.ipynb
aifr-face-rec.ipynb
best_age_model.pth
best_aifr_model.pth
deployment.py
Result.png
Result2.png
saeed_saleh.jpeg
MTL/
    mtl-aifr.ipynb
    train.py
    data/
        data_loader.py
    draft_notebooks/
        MTL_AIFR.ipynb
        mtl-aifr (2).ipynb
    losses/
        age_loss.py
        cosface_loss.py
    models/
        afd.py
        aifr.py
        backbone.py
        grl.py
        heads.py
```

- `deployment.py`: Main Streamlit app for deployment.
- `best_age_model.pth`, `best_aifr_model.pth`: Pre-trained model weights.
- `MTL/`: Multi-task learning code and supporting modules.

## Getting Started

### Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/)
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- [Streamlit](https://streamlit.io/)
- OpenCV, Pillow, NumPy

Install dependencies:
```sh
pip install torch torchvision facenet-pytorch streamlit opencv-python pillow numpy
```

### Running the App

1. Ensure `best_age_model.pth` and `best_aifr_model.pth` are in the project root.
2. Start the Streamlit app:
    ```sh
    streamlit run deployment.py
    ```
3. Open the provided local URL in your browser.

### Usage

- Upload two face images.
- The app will:
  - Detect and align faces.
  - Predict the age for each face.
  - Compute similarity and verify if both faces belong to the same person.

## Model Training

- Training scripts and notebooks are in the `MTL/` directory.
- See `MTL/train.py` and the Jupyter notebooks for details.

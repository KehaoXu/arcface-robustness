# ArcFace Robustness Analysis

This project studies how image qualities affect ArcFace face recognition embeddings.

## Files

- **arcface.ipynb**  
  Loads the ArcFace model, extracts face embeddings, computes cosine similarity, and visualizes the results.

- **downsample_and_noise.ipynb**  
  Generates degraded images by applying downsampling and Gaussian noise.

- **preprocessing.py**  
  Resize the original dataset into (512, 512).

## Others

Example test images are stored in `testimage/`, and results are saved in `results/`.
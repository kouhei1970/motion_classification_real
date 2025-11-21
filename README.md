# Hand Gesture Classification with Transformers

This project implements a hand gesture classification pipeline using MediaPipe for landmark extraction and a Transformer-based model for sequence classification.

## Project Structure

```
.
├── data/                   # Generated data (landmarks, labels) - Excluded from git
├── dataset/                # Raw video dataset and annotations - Videos excluded from git
│   ├── avi/                # Original AVI videos
│   ├── mp4/                # Converted MP4 videos
│   └── Annot_List.txt      # Annotation file
├── src/                    # Source code
│   ├── process_real_data.py # Data extraction script (MediaPipe)
│   ├── train.py            # Training script (Transformer)
│   ├── predict.py          # Inference and visualization script
│   ├── model.py            # Model definition
│   ├── dataset.py          # PyTorch Dataset definition
│   └── visualize.py        # Visualization utilities
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Processing

Extract landmarks from video files based on annotations (`dataset/Annot_List.txt`).

```bash
python src/process_real_data.py
```
This will generate `data/real_landmarks.npy` and `data/real_labels.npy`.

### 2. Training

Train the Transformer model.

```bash
python src/train.py
```

**Options:**
- Resume training from an existing model:
  ```bash
  python src/train.py --resume --epochs 50
  ```

The trained model will be saved as `gesture_model.pth`.
Training history (loss/accuracy) will be saved as `training_history.png`.

### 3. Prediction & Visualization

Run inference on the dataset and visualize the results with 3D skeletal animation.

```bash
python src/predict.py
```
- Use **Left/Right Arrow Keys** to switch between samples.
- The title shows the True Label and Predicted Label with confidence score.

## Model Architecture

- **Input**: Sequence of 100 frames. Each frame contains 21 hand landmarks (x, y, z).
- **Features**: Coordinates + Velocity + Acceleration (Total 189 dimensions per frame).
- **Model**: Transformer Encoder with Positional Encoding.
- **Output**: Class probabilities.

## Requirements

- Python 3.8+
- PyTorch
- MediaPipe
- OpenCV
- NumPy
- Matplotlib

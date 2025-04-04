# Long Audio Speech Emotion Classifier

This project provides a Python script to analyze long audio files using a pre-trained, TorchScript-exported speech emotion classifier model (e.g., classifying segments as 'calm' vs 'heated'). It processes the audio using a sliding window approach and outputs timestamped predictions to CSV files. Optional multi-stage smoothing can be applied to the raw predictions for more coherent results.

## Features

*   **Long Audio Processing:** Efficiently handles long audio files without loading the entire file into memory multiple times for processing.
*   **Sliding Window Inference:** Analyzes audio in overlapping segments.
*   **TorchScript Model Support:** Uses pre-trained models exported to TorchScript format (`.pt`) for optimized inference.
*   **Timestamped Output:** Generates CSV files mapping predicted labels ('calm', 'heated', 'uncertain') to specific time segments in the audio.
*   **Activation Threshold:** Classifies segments below a certain prediction probability threshold as 'uncertain'.
*   **Advanced Smoothing:** Optional multi-stage smoothing pipeline:
    *   Merges consecutive identical predictions.
    *   Bridges short segments between longer segments of the same class.
    *   Absorbs 'uncertain' and other short segments into neighbors.
    *   Performs a final merge pass.
*   **Modular Code:** Organized into separate Python modules for configuration, audio processing, inference, smoothing, utilities, and logging.
*   **CPU/GPU Support:** Can run inference on either CPU or CUDA-enabled GPU.

## Prerequisites

*   Python 3.8+
*   Conda package and environment manager

## Installation

It is **highly recommended** to use a Conda environment (Method 2) to avoid dependency conflicts. Direct installation (Method 1) is not recommended for general use.

*(Assumes you have already cloned the repository and navigated into the project directory.)*

---

### Method 1: Direct pip Install (Not Recommended)

**Warning:** This installs packages globally or into your user site-packages and can lead to dependency conflicts between projects.

1.  **Install PyTorch & Torchaudio:**
    Get the correct command for your OS/CUDA setup from the [Official PyTorch Website](https://pytorch.org/get-started/locally/) (select `pip` package).

    *   *Example (CPU only):*
        ```bash
        pip install torch torchvision torchaudio
        ```
    *   *Example (CUDA 11.8):*
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
        *(Replace `cu118` with your CUDA version, e.g., `cu121`)*

2.  **Verify:**
    ```bash
    python -c "import torch; import torchaudio; print(f'Torch: {torch.__version__}, Torchaudio: {torchaudio.__version__}')"
    ```

---

### Method 2: Conda Environment (Recommended)

1.  **Create & Activate Environment:**
    ```bash
    conda create -n audio_classifier_env python=3.10 -y  # Or choose your Python version
    conda activate audio_classifier_env
    ```

2.  **Install PyTorch & Torchaudio:**
    Get the correct command for your OS/CUDA setup from the [Official PyTorch Website](https://pytorch.org/get-started/locally/) (select `conda` package).

    *   *Example (CPU only):*
        ```bash
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
        ```
    *   *Example (CUDA 11.8):*
        ```bash
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        ```
        *(Replace `pytorch-cuda=11.8` with your CUDA version, e.g., `pytorch-cuda=12.1`)*

3.  **Verify:**
    ```bash
    python -c "import torch; import torchaudio; print(f'Torch: {torch.__version__}, Torchaudio: {torchaudio.__version__}')"
    ```

---

Remember to activate the Conda environment (`conda activate audio_classifier_env`) before running the script if you used Method 2.

## Usage

You can run the classifier using one of the following methods:

### Option 1: Run the Script Directly

Use the `python` command to execute the `run_classifier.py` script. Replace the placeholders with the actual paths to your model and audio file.

```bash
python scripts/run_classifier.py --model /path/to/model.pt \
                                 --input /path/to/your_audio.wav \
                                 --output /path/to/output.csv \
                                 --window-sec 3.0 \
                                 --step-sec 1.0 \
                                 --activation-threshold 0.5 \
                                 --smoothing \
                                 --min-smooth-duration 1.5 \
                                 --device auto
```

### Option 2: Use the `run.sh` Script

A `run.sh` script is provided for convenience. Update the script with the paths to your model and audio file, or pass them as arguments if the script supports it.

1. Make the script executable (if not already):
    ```bash
    chmod +x run.sh
    ```

2. Run the script:
    ```bash
    ./run.sh
    ```

The `run.sh` script will internally call the `run_classifier.py` script with predefined or customizable arguments.

---

For both options, ensure that the required dependencies are installed and the environment (e.g., Conda or virtual environment) is activated if applicable.

## Arguments

The `run_classifier.py` script accepts the following command-line arguments:

| Argument                 | Required | Default                       | Description                                                                                                                                        |
| :----------------------- | :------- | :---------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model PATH`           | Yes      | N/A                           | Path to the exported TorchScript classifier model (`.pt` file).                                                                                    |
| `--input PATH`           | Yes      | N/A                           | Path to the input audio file (e.g., `.wav`, `.mp3`).                                                                                               |
| `--output PATH`          | No       | `output_timestamps_raw.csv` | Path for saving the detailed (raw, per-window) results CSV. If smoothing is enabled, smoothed results saved to `<output_path_stem>_smoothed.csv`. |
| `--window-sec FLOAT`     | No       | `3.0`                         | Duration of the analysis window in seconds.                                                                                                        |
| `--step-sec FLOAT`       | No       | `1.0`                         | Step size (overlap) of the sliding window in seconds.                                                                                              |
| `--activation-threshold FLOAT` | No | `0.0`                         | Minimum probability (0.0-1.0) for a class prediction. Below this, label is 'uncertain'. Set to `0.0` to disable.                                   |
| `--smoothing`            | No       | `False`                       | Enable the multi-stage smoothing pipeline.                                                                                                         |
| `--min-smooth-duration FLOAT` | No | `1.5`                         | Minimum duration (seconds) for a segment *not* to be considered 'short' during advanced smoothing. Only active if `--smoothing` is enabled.        |
| `--device {auto,cuda,cpu}` | No    | `auto`                        | Device to run inference on (`auto`, `cuda`, `cpu`).                                                                                                |
| `--verbose`, `-v`        | No       | `False`                       | Enable verbose (DEBUG level) logging.                                                                                                              |

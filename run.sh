#!/bin/bash

# --- Script Configuration ---
SCRIPT_PATH="scripts/run_classifier.py"

# --- Inference Arguments ---
MODEL_ARG="models/audio_classifier.pt"        # Path to the pre-trained TorchScript model file
INPUT_ARG="data/audio/youtube_audio.mp3"      # Path to the input audio file to be processed
OUTPUT_ARG="data/results/predictions.csv"     # Path for saving the raw, per-window prediction results CSV
WINDOW_SEC_ARG="3.0"                          # Duration (in seconds) of each analysis window
STEP_SEC_ARG="3.0"                            # Step size (in seconds) for the sliding window (no overlap here)
ACTIVATION_THRESHOLD_ARG="0.95"               # Minimum confidence probability (0.0-1.0) to accept prediction
MIN_SMOOTH_DURATION_ARG="5.0"                 # Min duration (sec) for segment merging during smoothing
DEVICE_ARG="auto"                             # Compute device (auto, cuda, cpu)

# --- Flags (set to the flag itself if needed, or ""/"#" if not) ---
# Enable multi-stage smoothing (creates *_smoothed.csv) - set to "--smoothing" or ""
SMOOTHING_FLAG="--smoothing"

# --- Execute the Command ---
echo "Starting inference with the following settings:"
echo "  Model: $MODEL_ARG"
echo "  Input: $INPUT_ARG"
echo "  Output (raw): $OUTPUT_ARG"
echo "  Window: ${WINDOW_SEC_ARG}s, Step: ${STEP_SEC_ARG}s"
echo "  Smoothing: $(if [ -n "$SMOOTHING_FLAG" ]; then echo "Enabled (Min Duration: ${MIN_SMOOTH_DURATION_ARG}s)"; else echo "Disabled"; fi)"
echo "  Activation Threshold: $ACTIVATION_THRESHOLD_ARG"
echo "  Device: $DEVICE_ARG"
echo "---"

# Build the command arguments conditionally for the smoothing flag
cmd_args=(
    --model "$MODEL_ARG"
    --input "$INPUT_ARG"
    --output "$OUTPUT_ARG"
    --window-sec "$WINDOW_SEC_ARG"
    --step-sec "$STEP_SEC_ARG"
    # Only add smoothing flag and its dependent arg if the flag is set
    ${SMOOTHING_FLAG:+"$SMOOTHING_FLAG"}
    ${SMOOTHING_FLAG:+--min-smooth-duration} ${SMOOTHING_FLAG:+"$MIN_SMOOTH_DURATION_ARG"}
    --activation-threshold "$ACTIVATION_THRESHOLD_ARG"
    --device "$DEVICE_ARG"
)

# Execute python script with constructed arguments
python "$SCRIPT_PATH" "${cmd_args[@]}"


echo "---"
echo "Script finished."
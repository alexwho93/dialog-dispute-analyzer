"""
Main script to run inference on a long audio file using a pre-trained
audio classifier model, with optional smoothing.
Orchestrates loading, prediction, smoothing, and saving.
"""

import torch
import argparse
import logging
import time
from pathlib import Path

from config import InferenceConfig, UNCERTAIN_LABEL_DEFAULT
from audio_processing import AudioProcessor
from inference import AudioClassifier
from smoothing import apply_smoothing_pipeline
from utils import save_results_to_csv

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# --- Argument Parsing Setup ---
# setup_arg_parser function remains exactly the same as before
def setup_arg_parser() -> argparse.ArgumentParser:
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run inference on a long audio file using a TorchScript audio classifier, "
                    "output timestamped predictions, and optionally apply advanced smoothing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Arguments (Identical to previous version) ---
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the exported TorchScript model file (.pt)."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input audio file (e.g., .wav, .mp3)."
    )
    parser.add_argument(
        "--output", type=str, default="output_timestamps_raw.csv",
        help="Path for saving the detailed (raw, per-window) results CSV. "
             "If smoothing is enabled, smoothed results saved to '<output_path_stem>_smoothed.csv'."
    )
    parser.add_argument(
        "--window-sec", type=float, metavar="FLOAT", default=3.0,
        help="Duration of the analysis window in seconds."
    )
    parser.add_argument(
        "--step-sec", type=float, metavar="FLOAT", default=1.0,
        help="Step size (overlap) of the sliding window in seconds."
    )
    parser.add_argument(
        "--activation-threshold", type=float, metavar="[0.0-1.0]", default=0.0,
        help=f"Minimum probability for the winning class. Predictions below this threshold are labeled '{UNCERTAIN_LABEL_DEFAULT}'. Set to 0.0 to disable."
    )
    parser.add_argument(
        "--smoothing", action='store_true',
        help="Enable multi-stage smoothing (merge consecutive, bridge short segments, absorb uncertain/short, final merge)."
    )
    parser.add_argument(
        "--min-smooth-duration", type=float, metavar="FLOAT", default=1.5,
        help="Minimum duration (seconds) for a segment *not* to be considered 'short' during advanced smoothing (bridging/absorption). Only active if --smoothing is enabled."
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Device to run inference on ('auto', 'cuda', 'cpu')."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose (DEBUG) logging."
    )
    return parser


# --- Main Orchestration Pipeline ---
def run_inference_pipeline(config: InferenceConfig):
    """
    Orchestrates the full inference pipeline using AudioProcessor and AudioClassifier.

    Args:
        config: The validated InferenceConfig object.
    """
    pipeline_start_time = time.time()
    logger.info(f"Starting inference pipeline for: {config.input_path}")
    logger.info(f"Using device: {config.device}")
    logger.info(f"Window: {config.window_sec:.2f}s ({config.window_samples} samples), Step: {config.step_sec:.2f}s ({config.step_samples} samples)")
    # ... (logging of other config params is the same) ...
    if config.activation_threshold > 0.0: logger.info(f"Activation Threshold: {config.activation_threshold:.2f} (Label: '{config.uncertain_label}')")
    else: logger.info("Activation Threshold: Disabled")
    logger.info(f"Multi-stage Smoothing: {'Enabled' if config.smoothing_enabled else 'Disabled'}")
    if config.smoothing_enabled: logger.info(f"Minimum Segment Duration for Smoothing: {config.min_smooth_duration:.2f}s")


    try:
        # --- 1. Instantiate Processor and Classifier ---
        # Processor handles audio loading/preprocessing settings
        processor = AudioProcessor(config.target_sr, config.device)

        # Classifier handles model loading and prediction settings
        # Model loading happens during __init__
        classifier = AudioClassifier(config.model_path, config.device, config)
        # Note: If model loading fails, AudioClassifier.__init__ will raise an exception here

        # --- 2. Load and Preprocess Audio using Processor ---
        # The processor instance handles the details based on its init args
        waveform = processor.load_and_preprocess(config.input_path)
        if waveform is None:
            logger.error("Pipeline aborted due to audio preprocessing failure.")
            return

        # --- 3. Predict Segments using Classifier ---
        logger.info("Starting prediction...")
        predict_start_time = time.time()
        # The classifier instance uses its loaded model and config
        detailed_results = classifier.predict_segments(waveform)
        predict_duration = time.time() - predict_start_time
        logger.info(f"Prediction finished in {predict_duration:.2f} seconds.")

        if not detailed_results:
            logger.warning("Prediction yielded no results. Pipeline finished.")
            return

        # --- Calculate RTF --- (remains the same)
        try:
             audio_duration = detailed_results[-1]['end_time_s']
             if audio_duration > 0: rtf = predict_duration / audio_duration; logger.info(f"Approx. Audio Duration: {audio_duration:.2f}s. RTF (Prediction): {rtf:.3f}")
             else: logger.info("Could not calculate RTF (audio duration is zero or negative).")
        except (IndexError, KeyError, TypeError): logger.warning("Could not calculate RTF.")

        # --- 4. Save Raw Results --- (remains the same)
        raw_headers = [
            'start_time_s', 'end_time_s', 'predicted_label', 'raw_predicted_label',
            'max_probability', 'probability_calm', 'probability_heated', 'probabilities'
        ]
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        save_results_to_csv(detailed_results, config.output_path, raw_headers)

        # --- 5. Apply Smoothing Pipeline --- (remains the same)
        smoothed_results = apply_smoothing_pipeline(
            raw_results=detailed_results,
            smoothing_enabled=config.smoothing_enabled,
            min_duration_s=config.min_smooth_duration,
            uncertain_label=config.uncertain_label
        )

        # --- 6. Save Smoothed Results --- (remains the same)
        if config.smoothing_enabled:
            if smoothed_results and config.smoothed_output_path:
                smoothed_headers = ['start_time_s', 'end_time_s', 'predicted_label']
                config.smoothed_output_path.parent.mkdir(parents=True, exist_ok=True)
                save_results_to_csv(smoothed_results, config.smoothed_output_path, smoothed_headers)
            elif not smoothed_results: logger.warning("Smoothing enabled but no segments to save.")

        pipeline_duration = time.time() - pipeline_start_time
        logger.info(f"Inference pipeline completed successfully in {pipeline_duration:.2f} seconds.")

    # Error handling remains largely the same, but init errors are now caught here too
    except FileNotFoundError as e: logger.error(f"Pipeline Error: Required file not found. {e}")
    except ValueError as e: logger.error(f"Pipeline Error: Configuration or data issue. {e}")
    except RuntimeError as e: logger.error(f"Pipeline Error: PyTorch runtime issue. {e}", exc_info=True)
    except Exception as e: logger.error(f"An unexpected error occurred during the pipeline: {e}", exc_info=True)


# --- Script Entry Point ---
# This block remains exactly the same as the previous version
if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers: handler.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    selected_device = args.device
    if selected_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = torch.device(selected_device)
    logger.info(f"Selected device: {device}")

    try:
        config = InferenceConfig(
            model_path=args.model, input_path=args.input, output_path=args.output,
            window_sec=args.window_sec, step_sec=args.step_sec,
            activation_threshold=args.activation_threshold,
            smoothing_enabled=args.smoothing, min_smooth_duration=args.min_smooth_duration,
            device=device
        )
        config.validate()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during configuration setup: {e}", exc_info=True)
        exit(1)

    run_inference_pipeline(config)
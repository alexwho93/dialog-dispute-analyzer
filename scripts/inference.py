import torch
import math
import logging
import time
from typing import List, Dict, Any
from config import InferenceConfig 
from pathlib import Path

logger = logging.getLogger(__name__)

class AudioClassifier:
    """Encapsulates the audio classification model and prediction logic."""

    def __init__(self, model_path: Path, device: torch.device, config: InferenceConfig):
        """
        Initializes the AudioClassifier, loading the model.

        Args:
            model_path: Path to the TorchScript model file.
            device: The torch device to run the model on.
            config: The InferenceConfig object containing relevant parameters.

        Raises:
            FileNotFoundError: If the model file does not exist.
            Exception: If model loading fails.
        """
        self.device = device
        self.config = config # Store the whole config or just needed parts
        self.model = self._load_model(model_path)
        logger.debug(f"AudioClassifier initialized with model: {model_path}")

    def _load_model(self, model_path: Path) -> torch.jit.ScriptModule:
        """Loads the TorchScript model onto the specified device."""
        if not model_path.is_file():
             raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading TorchScript model from: {model_path}")
        model_load_start = time.time()
        try:
            # Load directly to the instance's device
            model = torch.jit.load(model_path, map_location=self.device)
            model.eval() # Set to evaluation mode immediately
            logger.info(f"Model loaded successfully in {time.time() - model_load_start:.2f} seconds.")
            return model
        except Exception as e:
            logger.error(f"Failed to load TorchScript model from {model_path}: {e}", exc_info=True)
            raise # Re-raise the exception after logging

    def predict_segments(self, waveform: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Performs sliding window inference on the preprocessed waveform using the loaded model.

        Args:
            waveform: The preprocessed 1D audio waveform tensor.

        Returns:
            A list of dictionaries representing predicted segments.
        """
        results = []
        if waveform is None or waveform.numel() == 0:
            logger.error("Cannot predict segments: Invalid waveform provided.")
            return results

        # Access parameters from the stored config object
        target_sr = self.config.target_sr
        window_samples = self.config.window_samples
        step_samples = self.config.step_samples
        activation_threshold = self.config.activation_threshold
        id2label = self.config.id2label
        uncertain_label = self.config.uncertain_label

        total_samples = waveform.shape[0]
        total_duration_s = total_samples / target_sr

        if total_samples < window_samples:
             logger.warning(f"Audio duration ({total_duration_s:.2f}s) is shorter than window size "
                            f"({self.config.window_sec:.2f}s). Processing as a single padded segment.")
             padding_needed = window_samples - total_samples
             # Use waveform device directly
             waveform_padded = torch.nn.functional.pad(waveform, (0, padding_needed))
             waveform_to_process = waveform_padded
             num_windows = 1
             effective_step_samples = window_samples
        else:
             num_windows = math.floor((total_samples - window_samples) / step_samples) + 1
             waveform_to_process = waveform
             effective_step_samples = step_samples

        logger.info(f"Processing {num_windows} windows...")
        inference_start_time = time.time()

        for i in range(num_windows):
            start_sample = i * effective_step_samples
            end_sample_ts = min(start_sample + window_samples, total_samples)
            segment_slice = waveform_to_process[start_sample : start_sample + window_samples]

            current_segment_len = segment_slice.shape[0]
            if current_segment_len < window_samples:
                padding_needed = window_samples - current_segment_len
                segment = torch.nn.functional.pad(segment_slice, (0, padding_needed))
            else:
                segment = segment_slice

            if segment.ndim == 0 or segment.numel() == 0:
                logger.warning(f"Encountered empty segment at window {i}, skipping.")
                continue

            input_waveform = segment.unsqueeze(0).to(self.device) # Ensure on correct device

            try:
                # Use the instance's model
                with torch.no_grad():
                    logits = self.model(input_waveform)

                probabilities = torch.softmax(logits, dim=-1)[0]
                predicted_id = torch.argmax(probabilities).item()
                max_prob = probabilities[predicted_id].item()
                raw_predicted_label = id2label.get(predicted_id, f"Unknown ID: {predicted_id}")

                if max_prob >= activation_threshold:
                    final_predicted_label = raw_predicted_label
                else:
                    final_predicted_label = uncertain_label

                probs_dict = {id2label.get(j, f"Unknown ID: {j}"): prob.item()
                              for j, prob in enumerate(probabilities)}

                start_time_s = start_sample / target_sr
                end_time_s = end_sample_ts / target_sr

                results.append({
                    'start_time_s': start_time_s, 'end_time_s': end_time_s,
                    'predicted_label': final_predicted_label,
                    'raw_predicted_label': raw_predicted_label,
                    'max_probability': max_prob, 'probabilities': probs_dict,
                    'probability_calm': probs_dict.get('calm', 0.0),
                    'probability_heated': probs_dict.get('heated', 0.0)
                })

                if (i + 1) % 50 == 0 or (i + 1) == num_windows:
                    logger.info(f"Processed window {i+1}/{num_windows} (up to {end_time_s:.2f}s)")

            except Exception as e:
                logger.error(f"Error during inference for window {i} (start: {start_sample}): {e}", exc_info=True)
                continue

        inference_duration = time.time() - inference_start_time
        logger.info(f"Finished processing {num_windows} windows ({inference_duration:.2f}s).")
        return results
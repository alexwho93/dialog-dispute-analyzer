import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from typing import Optional, Dict
import logging
import time

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles loading, preprocessing (resampling, mono), and caches resamplers."""

    def __init__(self, target_sr: int, device: torch.device):
        """
        Initializes the AudioProcessor.

        Args:
            target_sr: The target sample rate for audio.
            device: The torch device to use for processing.
        """
        self.target_sr = target_sr
        self.device = device
        self._resampler_cache: Dict[tuple, T.Resample] = {} # Instance cache
        logger.debug(f"AudioProcessor initialized for SR={target_sr} on device={device}")

    def _get_resampler(self, source_sr: int) -> T.Resample:
        """Gets or creates a resampler for the instance's target SR and device."""
        # Target SR and device are now instance attributes
        key = (source_sr, self.target_sr, str(self.device))
        if key not in self._resampler_cache:
            logger.info(f"Creating resampler for {source_sr} Hz -> {self.target_sr} Hz on {self.device}...")
            self._resampler_cache[key] = T.Resample(
                orig_freq=source_sr, new_freq=self.target_sr
            ).to(self.device)
        return self._resampler_cache[key]

    def load_and_preprocess(self, audio_path: Path) -> Optional[torch.Tensor]:
        """
        Loads audio, resamples, converts to mono using instance settings.

        Args:
            audio_path: Path to the input audio file.

        Returns:
            A 1D torch.Tensor containing the preprocessed waveform, or None on error.
        """
        logger.info(f"Loading and preprocessing audio: {audio_path}")
        try:
            load_start_time = time.time()
            # Load directly to the instance's device
            waveform_full, sr = torchaudio.load(audio_path)
            waveform_full = waveform_full.to(self.device)
            logger.info(f"Loaded audio ({time.time() - load_start_time:.2f}s). Original SR: {sr} Hz, Shape: {waveform_full.shape}")

            # Resample if needed using instance's target SR
            if sr != self.target_sr:
                resample_start_time = time.time()
                resampler = self._get_resampler(sr) # Use internal method
                waveform_full = resampler(waveform_full)
                logger.info(f"Resampled to {self.target_sr} Hz ({time.time() - resample_start_time:.2f}s). New Shape: {waveform_full.shape}")

            # Convert to mono if needed
            if waveform_full.shape[0] > 1:
                mono_start_time = time.time()
                waveform_full = waveform_full.mean(dim=0, keepdim=False)
                logger.info(f"Converted to mono ({time.time() - mono_start_time:.2f}s). New Shape: {waveform_full.shape}")
            elif waveform_full.ndim > 1:
                waveform_full = waveform_full.squeeze(0)

            total_samples = waveform_full.shape[0]
            if total_samples == 0:
                logger.error("Audio has zero length after preprocessing.")
                return None

            total_duration_s = total_samples / self.target_sr
            logger.info(f"Total duration after preprocessing: {total_duration_s:.2f} seconds.")
            return waveform_full

        except FileNotFoundError:
            logger.error(f"Audio file not found: {audio_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading/preprocessing {audio_path}: {e}", exc_info=True)
            return None
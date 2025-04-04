import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List
import logging
import os
import argparse # Keep for default checking in validate

logger = logging.getLogger(__name__)

# --- Configuration Constants ---
TARGET_SAMPLE_RATE = 16000
ID2LABEL_DEFAULT = {0: "calm", 1: "heated"}
UNCERTAIN_LABEL_DEFAULT = "uncertain"

@dataclass
class InferenceConfig:
    """Holds configuration parameters for the inference pipeline."""
    model_path: Path
    input_path: Path
    output_path: Path # Base path for raw output csv
    window_sec: float
    step_sec: float
    activation_threshold: float
    smoothing_enabled: bool
    min_smooth_duration: float
    device: torch.device
    target_sr: int = TARGET_SAMPLE_RATE
    id2label: Dict[int, str] = field(default_factory=lambda: ID2LABEL_DEFAULT)
    uncertain_label: str = UNCERTAIN_LABEL_DEFAULT
    # Derived properties
    window_samples: int = field(init=False)
    step_samples: int = field(init=False)
    smoothed_output_path: Optional[Path] = field(init=False, default=None)

    def __post_init__(self):
        """Calculate derived values after initialization."""
        self.window_samples = int(self.window_sec * self.target_sr)
        self.step_samples = int(self.step_sec * self.target_sr)
        # Ensure paths are Path objects
        self.model_path = Path(self.model_path)
        self.input_path = Path(self.input_path)
        self.output_path = Path(self.output_path)
        if self.smoothing_enabled:
             self.smoothed_output_path = self.output_path.parent / f"{self.output_path.stem}_smoothed{self.output_path.suffix}"
        logger.debug(f"Config initialized: {self}")

    def validate(self) -> bool:
        """Performs validation checks on the configuration."""
        logger.info("Validating configuration...")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Input audio file not found: {self.input_path}")
        if self.window_sec <= 0:
            raise ValueError("Window duration must be positive.")
        if self.step_sec <= 0:
            raise ValueError("Step duration must be positive.")
        if not (0.0 <= self.activation_threshold <= 1.0):
            raise ValueError("Activation threshold must be between 0.0 and 1.0.")
        if self.smoothing_enabled and self.min_smooth_duration <= 0:
             raise ValueError("--min-smooth-duration must be positive when --smoothing is enabled.")

        # Non-fatal warnings
        if self.step_sec > self.window_sec:
            logger.warning("Step duration > window duration; parts of audio will be skipped.")
        if not self.smoothing_enabled:
            # Check if min_smooth_duration was explicitly set by user when smoothing is off
             try:
                 # Need to get defaults - slightly awkward here without passing parser
                 # We'll rely on the main script to potentially re-log this warning
                 # Or pass the default value if needed more robustly.
                 # Simple check based on typical default:
                 if self.min_smooth_duration != 1.5: # Assuming 1.5 is the default
                     user_provided_min_smooth = False
                     for arg in os.sys.argv: # Check command line args passed
                         if arg.startswith('--min-smooth-duration'): user_provided_min_smooth = True; break
                     if user_provided_min_smooth:
                          logger.warning("--min-smooth-duration is set but --smoothing is not enabled. It will have no effect.")
             except Exception: # Catch any error during this check
                 pass

        logger.info("Configuration validated successfully.")
        return True
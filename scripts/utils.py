import csv
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def save_results_to_csv(results: List[Dict[str, Any]], output_path: Path, headers: List[str]):
    """
    Saves a list of result dictionaries to a CSV file.

    Args:
        results: List of dictionaries (each dictionary is a row).
        output_path: Path object for the output CSV file.
        headers: List of strings defining the CSV column headers and order.
    """
    # (Implementation is identical to the previous full script version)
    if not results:
        logger.warning(f"No results provided to save to {output_path}.")
        return
    if not headers:
         logger.error(f"No headers provided for saving CSV {output_path}.")
         return
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving {len(results)} results to: {output_path}")
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            for row_dict in results:
                formatted_row = {}
                for key in headers:
                    value = row_dict.get(key)
                    if isinstance(value, float):
                        try: formatted_row[key] = f"{value:.4f}"
                        except (ValueError, TypeError):
                             logger.warning(f"Could not format float value for key '{key}': {value}")
                             formatted_row[key] = str(value)
                    elif value is not None: formatted_row[key] = value
                writer.writerow(formatted_row)
        logger.info(f"Successfully saved results to: {output_path}")
    except IOError as e:
         logger.error(f"Failed to open or write to CSV file {output_path}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving results to {output_path}: {e}", exc_info=True)
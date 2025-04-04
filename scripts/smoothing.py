import logging
from typing import List, Dict, Any
# Note: No direct dependency on InferenceConfig here, pass necessary params explicitly

logger = logging.getLogger(__name__)

# --- Smoothing Functions ---

def merge_consecutive_predictions(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merges consecutive windows with the same 'predicted_label'."""
    # (Implementation is identical to the previous full script version)
    if not results: return []
    merged_segments = []
    if len(results) == 1:
         res = results[0]
         merged_segments.append({
             'start_time_s': res['start_time_s'],
             'end_time_s': res['end_time_s'],
             'predicted_label': res['predicted_label']
         })
         return merged_segments
    current_segment = {
        'start_time_s': results[0]['start_time_s'],
        'end_time_s': results[0]['end_time_s'],
        'predicted_label': results[0]['predicted_label']
    }
    for i in range(1, len(results)):
        window_result = results[i]; label = window_result['predicted_label']
        start = window_result['start_time_s']; end = window_result['end_time_s']
        if label == current_segment['predicted_label']:
            current_segment['end_time_s'] = max(current_segment['end_time_s'], end)
        else:
            merged_segments.append(current_segment)
            current_segment = {'start_time_s': start, 'end_time_s': end, 'predicted_label': label}
        if i == len(results) - 1: merged_segments.append(current_segment)
    logger.info(f"Initial merging reduced {len(results)} windows to {len(merged_segments)} segments.")
    return merged_segments


def apply_advanced_smoothing(
    segments: List[Dict[str, Any]],
    min_segment_duration_s: float,
    uncertain_label: str # Pass uncertain label explicitly
) -> List[Dict[str, Any]]:
    """Applies advanced smoothing rules (bridging, absorbing uncertain/short)."""
    # (Implementation is identical to the previous full script version,
    # just ensure uncertain_label is used from the argument)
    if not segments or min_segment_duration_s <= 0:
        logger.debug("Skipping advanced smoothing (no segments or min duration <= 0).")
        return segments
    logger.info(f"Applying advanced smoothing rules (min duration: {min_segment_duration_s:.2f}s)...")
    smoothed = []
    num_segments = len(segments)
    i = 0
    processed_indices = set()
    while i < num_segments:
        if i in processed_indices: i += 1; continue
        current_segment = segments[i]
        current_duration = current_segment['end_time_s'] - current_segment['start_time_s']
        is_short = current_duration < min_segment_duration_s
        prev_segment = segments[i - 1] if i > 0 else None
        next_segment = segments[i + 1] if i + 1 < num_segments else None
        # Rule 1: Bridge
        can_bridge = (is_short and prev_segment and next_segment and
                      prev_segment['predicted_label'] == next_segment['predicted_label'] and
                      current_segment['predicted_label'] != prev_segment['predicted_label'] and
                      prev_segment['predicted_label'] != uncertain_label)
        if can_bridge:
            merged_label = prev_segment['predicted_label']; merged_start = prev_segment['start_time_s']
            merged_end = next_segment['end_time_s']
            logger.info(f"Bridging short segment {i} ('{current_segment['predicted_label']}', {current_duration:.2f}s) "
                        f"between '{merged_label}' segments. New span: {merged_start:.2f}s - {merged_end:.2f}s.")
            if smoothed and smoothed[-1]['end_time_s'] == prev_segment['end_time_s']:
                 smoothed[-1]['end_time_s'] = merged_end; smoothed[-1]['predicted_label'] = merged_label
            else:
                 logger.warning(f"Bridging segment {i} required creating a new merged entry.")
                 smoothed.append({'start_time_s': merged_start, 'end_time_s': merged_end, 'predicted_label': merged_label})
            processed_indices.add(i); processed_indices.add(i + 1); i += 2; continue
        # Rule 2: Absorb Uncertain
        is_uncertain = current_segment['predicted_label'] == uncertain_label
        if is_uncertain:
            merged = False
            if smoothed:
                 logger.info(f"Absorbing uncertain segment {i} ({current_duration:.2f}s) into previous ('{smoothed[-1]['predicted_label']}').")
                 smoothed[-1]['end_time_s'] = max(smoothed[-1]['end_time_s'], current_segment['end_time_s'])
                 processed_indices.add(i); i += 1; merged = True; continue
            elif next_segment:
                 logger.info(f"Absorbing first uncertain segment {i} ({current_duration:.2f}s) into next ('{next_segment['predicted_label']}').")
                 merged_segment = {'start_time_s': current_segment['start_time_s'], 'end_time_s': next_segment['end_time_s'], 'predicted_label': next_segment['predicted_label']}
                 smoothed.append(merged_segment)
                 processed_indices.add(i); processed_indices.add(i + 1); i += 2; merged = True; continue
        # Rule 3: Absorb Other Short
        if is_short and not is_uncertain:
             merged = False
             if smoothed:
                 logger.info(f"Absorbing short segment {i} ('{current_segment['predicted_label']}', {current_duration:.2f}s) into previous ('{smoothed[-1]['predicted_label']}').")
                 smoothed[-1]['end_time_s'] = max(smoothed[-1]['end_time_s'], current_segment['end_time_s'])
                 processed_indices.add(i); i += 1; merged = True; continue
             elif next_segment:
                  logger.info(f"Absorbing first short segment {i} ('{current_segment['predicted_label']}', {current_duration:.2f}s) into next ('{next_segment['predicted_label']}').")
                  merged_segment = {'start_time_s': current_segment['start_time_s'], 'end_time_s': next_segment['end_time_s'], 'predicted_label': next_segment['predicted_label']}
                  smoothed.append(merged_segment)
                  processed_indices.add(i); processed_indices.add(i + 1); i += 2; merged = True; continue
        # Default: Add Segment
        if i not in processed_indices:
             if not smoothed or smoothed[-1]['end_time_s'] <= current_segment['start_time_s']:
                  smoothed.append(current_segment.copy())
             elif smoothed:
                  logger.warning(f"Skipping segment {i} due to potential overlap after merge. Prev end: {smoothed[-1]['end_time_s']:.2f}, Curr start: {current_segment['start_time_s']:.2f}")
             processed_indices.add(i)
        i += 1
    logger.info(f"Advanced smoothing resulted in {len(smoothed)} segments.")
    return smoothed


def merge_final_smoothed_segments(
    segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Performs a final pass to merge adjacent segments with the same label."""
    # (Implementation is identical to the previous full script version)
    if not segments: return []
    logger.info(f"Applying final merge pass to {len(segments)} segments...")
    final_merged_segments = []
    if len(segments) == 1: return segments
    current_segment = segments[0].copy()
    for i in range(1, len(segments)):
        next_segment = segments[i]
        if next_segment['predicted_label'] == current_segment['predicted_label']:
            current_segment['end_time_s'] = max(current_segment['end_time_s'], next_segment['end_time_s'])
        else:
            final_merged_segments.append(current_segment)
            current_segment = next_segment.copy()
        if i == len(segments) - 1: final_merged_segments.append(current_segment)
    final_count = len(final_merged_segments); initial_count = len(segments)
    if initial_count > final_count: logger.info(f"Final merge reduced {initial_count} segments to {final_count}.")
    else: logger.info(f"  No adjacent segments to merge in final pass.")
    return final_merged_segments


# --- Combined Smoothing Pipeline ---
def apply_smoothing_pipeline(
    raw_results: List[Dict[str, Any]],
    smoothing_enabled: bool,           # Pass explicitly
    min_duration_s: float,           # Pass explicitly
    uncertain_label: str             # Pass explicitly
) -> List[Dict[str, Any]]:
    """
    Applies the configured smoothing pipeline to raw prediction results.
    Uses explicitly passed parameters instead of the full config object.
    """
    if not raw_results:
        logger.warning("No raw results to smooth.")
        return []

    # Always perform the initial merge for basic cleanup if results exist
    merged_stage1 = merge_consecutive_predictions(raw_results)

    if not smoothing_enabled:
        logger.info("Advanced smoothing disabled. Returning initially merged segments.")
        return merged_stage1

    logger.info("Applying full multi-stage smoothing pipeline...")
    # Stage 2: Apply advanced rules
    smoothed_stage2 = apply_advanced_smoothing(merged_stage1, min_duration_s, uncertain_label)
    # Stage 3: Final merge pass
    final_smoothed_results = merge_final_smoothed_segments(smoothed_stage2)

    logger.info(f"Smoothing pipeline complete. Final segment count: {len(final_smoothed_results)}")
    return final_smoothed_results
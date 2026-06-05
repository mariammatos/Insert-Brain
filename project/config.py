# ============================================================
# FILE: config.py
#
# Central configuration for the BCI experiment.
# Defines board settings, channel names, class labels,
# trial timings, and baseline durations.
# ============================================================

from brainflow.board_shim import BoardIds

CONFIG = {

    "participant_id": "P001",

    "channel_names": ["F3", "F4", "FC4", "C3", "FCz", "CP3", "CP4", "CPz"],


    # BOARD CONFIGURATION
    "board_id": BoardIds.CYTON_BOARD.value,
    "serial_port": "COM8",

    # EXPERIMENT SETTINGS
    "classes": {
        "LEFT":  {"label": 1, "symbol": "←", "trials": 5},
        "RIGHT": {"label": 2, "symbol": "→", "trials": 5},
        "FEET":  {"label": 3, "symbol": "↓", "trials": 10},
        "REST":  {"label": 0, "symbol": "+", "trials": 20},
    },

    # TIMINGS (SECONDS)
    "fixation_time":  2.0,
    "cue_time":       2.0,
    "imagery_time":   5.0,
    "rest_time":      2.0,

    # BASELINE
    "baseline_open":   15,
    "baseline_closed": 15
}
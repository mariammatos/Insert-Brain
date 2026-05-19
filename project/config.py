# ============================================================
# FILE: config.py
# ============================================================

from brainflow.board_shim import BoardIds

CONFIG = {

    # ========================================================
    # PARTICIPANT
    # ========================================================

    "participant_id": "P001",

    # ========================================================
    # BOARD CONFIGURATION
    # ========================================================

    # FOR TESTING WITHOUT HARDWARE
    "board_id": BoardIds.SYNTHETIC_BOARD.value,

    # FOR REAL OPENBCI LATER:
    # "board_id": BoardIds.CYTON_BOARD.value,

    "serial_port": "COM3",

    # ========================================================
    # EXPERIMENT SETTINGS
    # ========================================================

    "trials_per_class": 10,

    "classes": {

        "LEFT": {
            "label": 1,
            "symbol": "←"
        },

        "RIGHT": {
            "label": 2,
            "symbol": "→"
        },

        "FEET": {
            "label": 3,
            "symbol": "↓"
        },

        # BASELINE / REST
        "REST": {
            "label": 0,
            "symbol": "+"
        }
    },

    # ========================================================
    # TIMINGS (SECONDS)
    # ========================================================

    "fixation_time": 2.0,
    "cue_time": 1.0,
    "imagery_time": 4.0,
    "rest_time": 2.0,

    # ========================================================
    # BASELINE
    # ========================================================

    "baseline_open": 30,
    "baseline_closed": 30
}
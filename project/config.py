# ============================================================
# FILE: config.py
# ============================================================

from brainflow.board_shim import BoardIds

CONFIG = {

    "participant_id": "P001",

    # ========================================================
    # BOARD CONFIGURATION
    # ========================================================

    "board_id": BoardIds.CYTON_BOARD.value,  # 8 canais
    "serial_port": "COM3",

    # ========================================================
    # EXPERIMENT SETTINGS
    # ========================================================

    "trials_per_class": 10,

    "classes": {
        "LEFT":  {"label": 1, "symbol": "←"},
        "RIGHT": {"label": 2, "symbol": "→"},
        "FEET":  {"label": 3, "symbol": "↓"},
        "REST":  {"label": 0, "symbol": "+"}
    },

    # ========================================================
    # TIMINGS (SECONDS)
    # ========================================================

    "fixation_time":  2.0,
    "cue_time":       2.0,
    "imagery_time":   5.0,
    "rest_time":      2.0,

    # ========================================================
    # BASELINE
    # ========================================================

    "baseline_open":   30,
    "baseline_closed": 30
}
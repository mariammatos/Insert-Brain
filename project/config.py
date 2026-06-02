# ============================================================
# FILE: config.py
# ============================================================

from brainflow.board_shim import BoardIds

CONFIG = {

    "participant_id": "P001",

    # WE STILL NEED TO CHANGE THIS IN OTHER FILES, EACH FILE HAS SPECIFIC CHANNELS
    "channel_names": ["F3", "F4", "C3", "FC4", "FCz", "CP3", "CP4", "CPz"], # don't forget the gel on GND and REF!!


    # ========================================================
    # BOARD CONFIGURATION
    # ========================================================

    "board_id": BoardIds.CYTON_BOARD.value,  # 8 canais
    "serial_port": "COM13",

    # ========================================================
    # EXPERIMENT SETTINGS
    # ========================================================

    "classes": {
        "LEFT":  {"label": 1, "symbol": "←", "trials": 5},
        "RIGHT": {"label": 2, "symbol": "→", "trials": 5},
        "FEET":  {"label": 3, "symbol": "↓", "trials": 10},
        "REST":  {"label": 0, "symbol": "+", "trials": 20},
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

    "baseline_open":   15,
    "baseline_closed": 15
}
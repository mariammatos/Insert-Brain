# ============================================================
# FILE: main.py
# ============================================================

from psychopy import prefs

prefs.hardware['audioLib'] = ['PTB']

from config import CONFIG

from acquisition.eeg_stream import EEGStream
from acquisition.data_logger import DataLogger

from protocol.protocol import MotorImageryProtocol


# ============================================================
# LOGGER
# ============================================================

logger = DataLogger(
    CONFIG["participant_id"]
)

# ============================================================
# EEG STREAM
# ============================================================

stream = EEGStream(CONFIG)

stream.start()

# ============================================================
# RUN PROTOCOL
# ============================================================

protocol = MotorImageryProtocol(
    CONFIG,
    logger
)

protocol.run()

# ============================================================
# GET DATA BEFORE STOP
# ============================================================

raw_data = stream.get_data()

# ============================================================
# STOP STREAM
# ============================================================

stream.stop()

# ============================================================
# SAVE EEG
# ============================================================

logger.save_eeg(
    raw_data,
    stream.eeg_channels,
    stream.timestamp_channel
)

# ============================================================
# SAVE MARKERS
# ============================================================

logger.save_markers()

# ============================================================
# SAVE METADATA
# ============================================================

metadata = CONFIG.copy()

metadata["sampling_rate"] = (
    stream.get_sampling_rate()
)

metadata["eeg_channels"] = (
    stream.eeg_channels
)

logger.save_metadata(metadata)

print("\nSession complete.")
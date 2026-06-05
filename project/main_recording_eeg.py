# ============================================================
# FILE: main_recording_eeg.py
#
# Runs full EEG session pipeline:
# - starts EEG stream
# - runs Motor Imagery protocol
# - saves session data
# - optionally trains subject-specific models
# ============================================================

from psychopy import prefs

prefs.hardware['audioLib'] = ['PTB']

from config import CONFIG
from acquisition.eeg_stream import EEGStream
from acquisition.data_logger import DataLogger
from protocol.protocol import MotorImageryProtocol
from training.train_subject_model import train_subject_model


# ============================================================
# SESSION SETUP
# ============================================================

participant_id = input("Participant ID: ")

logger = DataLogger(participant_id)


# ============================================================
# START EEG STREAM
# ============================================================

stream = EEGStream(CONFIG)
stream.start()


# ============================================================
# RUN EXPERIMENT PROTOCOL
# ============================================================

try:
    protocol = MotorImageryProtocol(CONFIG, logger)
    protocol.run()

finally:
    # Ensure safe shutdown and data capture

    eeg_channels = stream.eeg_channels
    timestamp_channel = stream.timestamp_channel
    sampling_rate = stream.get_sampling_rate()

    raw_data = stream.get_data()
    stream.stop()

    if raw_data is not None:
        logger.save_eeg(
            raw_data,
            eeg_channels,
            timestamp_channel
        )

        logger.save_markers()

        metadata = CONFIG.copy()
        metadata["sampling_rate"] = sampling_rate
        metadata["eeg_channels"] = eeg_channels

        logger.save_metadata(metadata)

        print("Session completed successfully.")
    else:
        print("Warning: No EEG data received — session not saved.")


# ============================================================
# TRAIN SUBJECT MODEL
# ============================================================

if raw_data is not None:
    print("Training subject-specific model...")
    clf_gate, clf_axis, clf_dir, report = train_subject_model(
        logger.session_path
    )
else:
    print("Training skipped — no EEG data available.")


# ============================================================
# END
# ============================================================

print(f"Models saved at: {logger.session_path}")
print("Session complete.")
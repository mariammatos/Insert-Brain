# ============================================================
# FILE: acquisition/data_logger.py
# ============================================================

import os
import json
import pandas as pd
import time

from datetime import datetime


class DataLogger:

    def __init__(self, participant_id):

        # Timestamp legível — gera pastas como "P001_20250522_143000"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.session_path = os.path.join(
            "data",
            f"{participant_id}_{timestamp}"
        )

        os.makedirs(
            self.session_path,
            exist_ok=True
        )

        self.markers = []

    def add_marker(self, timestamp, event, label):

        self.markers.append({
            "timestamp": timestamp,
            "event":     event,
            "label":     label
        })

    def save_markers(self):

        df   = pd.DataFrame(self.markers)
        path = os.path.join(self.session_path, "markers.csv")
        df.to_csv(path, index=False)
        print(f"Markers saved: {path}")

    def save_eeg(self, data, eeg_channels, timestamp_channel):

        # Validação — evita crash silencioso se não houver dados
        if data is None or data.shape[1] == 0:
            print("AVISO: sem dados EEG para guardar.")
            return

        eeg        = data[eeg_channels, :].T
        timestamps = data[timestamp_channel, :]

        columns = [f"ch_{i}" for i in range(len(eeg_channels))]

        df = pd.DataFrame(eeg, columns=columns)
        df["timestamp"] = timestamps

        path = os.path.join(self.session_path, "eeg_raw.csv")
        df.to_csv(path, index=False)
        print(f"EEG saved: {path}")

    def save_metadata(self, metadata):

        path = os.path.join(self.session_path, "metadata.json")

        with open(path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Metadata saved: {path}")

# ============================================================
# FILE: acquisition/eeg_stream.py
# ============================================================

import time

from brainflow.board_shim import (
    BoardShim,
    BrainFlowInputParams
)


class EEGStream:

    def __init__(self, config):

        self.config = config

        params = BrainFlowInputParams()

        params.serial_port = config["serial_port"]

        self.board = BoardShim(
            config["board_id"],
            params
        )

        self.board_id = config["board_id"]

        self.eeg_channels = BoardShim.get_eeg_channels(
            self.board_id
        )

        self.timestamp_channel = BoardShim.get_timestamp_channel(
            self.board_id
        )

    def start(self):

        print("\nPreparing BrainFlow session...")

        self.board.prepare_session()

        print("Starting EEG stream...")

        self.board.start_stream()

        time.sleep(2)

        print("EEG stream started.\n")

    def stop(self):

        print("\nStopping EEG stream...")

        self.board.stop_stream()

        self.board.release_session()

        print("Session released.\n")

    def get_data(self):

        return self.board.get_board_data()

    def get_sampling_rate(self):

        return BoardShim.get_sampling_rate(
            self.board_id
        )
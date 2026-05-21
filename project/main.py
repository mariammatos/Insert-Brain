# ============================================================
# FILE: main.py
# ============================================================

from psychopy import prefs

prefs.hardware['audioLib'] = ['PTB']

from config import CONFIG

from acquisition.eeg_stream import EEGStream
from acquisition.data_logger import DataLogger

from protocol.protocol import MotorImageryProtocol

from training.train_subject_model import train_subject_model


# ============================================================
# LOGGER
# ============================================================

participant_id = input('Participant:')

logger = DataLogger(
    participant_id
)

# ============================================================
# EEG STREAM
# ============================================================

stream = EEGStream(CONFIG)

stream.start()

# ============================================================
# RUN PROTOCOL
# ============================================================

try:

    protocol = MotorImageryProtocol(
        CONFIG,
        logger
    )

    protocol.run()

finally:

    # Guarda propriedades ANTES de fechar qualquer coisa
    eeg_channels      = stream.eeg_channels
    timestamp_channel = stream.timestamp_channel
    sampling_rate     = stream.get_sampling_rate()

    # PRIMEIRO dados, DEPOIS fechar
    raw_data = stream.get_data()
    stream.stop()

    # Só guarda se tiver dados
    if raw_data is not None:

        logger.save_eeg(
            raw_data,
            eeg_channels,
            timestamp_channel
        )

        logger.save_markers()

        metadata = CONFIG.copy()
        metadata["sampling_rate"] = sampling_rate
        metadata["eeg_channels"]  = eeg_channels
        logger.save_metadata(metadata)

        print("\nAquisição completa.")

    else:
        print("\nAVISO: Nenhum dado EEG recebido — sessão não guardada.")

# ============================================================
# TRAIN SUBJECT MODEL
# ============================================================

if raw_data is not None:
    print("\nA iniciar treino do modelo...")
    clf_gate, clf_axis, clf_dir, report = train_subject_model(logger.session_path)
else:
    print("\nTreino ignorado — sem dados EEG.")

# ============================================================
# DONE
# ============================================================

print(
    f"\nModelos guardados em: {logger.session_path}"
)

print("\nSessão completa.")
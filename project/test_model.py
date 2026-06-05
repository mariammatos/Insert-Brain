# ============================================================
# FILE: test_model.py
#
# Real-time model testing for motor imagery EEG.
# Runs trials, classifies EEG windows, and collects accuracy
# metrics with optional user validation.
#
# Usage:
#   python test_model.py <session_path>
#   python test_model.py data/P001_20250521_143000
#
# Preprocessing matches training pipeline:
#   1. Average reference
#   2. Bandpass FIR 8–30 Hz
#   3. Notch filters (25 Hz, 50 Hz)
#   4. 4s classification window
#   5. CSP → Scaler → LDA pipeline (loaded from .pkl)
# ============================================================

import os
import sys
import time
import joblib
import numpy as np
from scipy.signal import firwin, sosfiltfilt, iirnotch

from brainflow.board_shim import BoardShim, BrainFlowInputParams
from config import CONFIG


# ============================================================
# CONFIG
# ============================================================

L_FREQ = 8.0
H_FREQ = 30.0
NOTCH_FREQS = [25.0, 50.0]

# Classification window = EPOCH_TMAX - EPOCH_TMIN from training
WINDOW_SEC = 4.0   # 4.5 - 0.5

# Extra padding so the filter does not have edge artifacts
# in the region of interest. Must be >= FIR filter length / sfreq.
# 2 s is safe for the filters used by MNE at 250 Hz.
FILTER_PAD_SEC = 2.0

# Preparation time before each trial (countdown)
PREP_SEC = 5

SYMBOLS = {
    0: "  +  ",
    1: "  ←  ",
    2: "  →  ",
    3: "  ↓  ",
}

NAMES = {
    0: "REST",
    1: "LEFT",
    2: "RIGHT",
    3: "FEET",
}

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ============================================================
# PREPROCESSING
# ============================================================

def make_fir_bandpass(l_freq, h_freq, sfreq):
    """
    Create FIR bandpass coefficients with firwin, matching MNE defaults.
    Returns b coefficients for use with sosfiltfilt via conversion.

    MNE uses fir_design="firwin" with automatic transition bandwidth.
    Here we replicate that with a conservative filter length (same as MNE
    uses for 250 Hz signals with these cutoff parameters).
    """
    # Filter length: MNE typically uses ~0.34 s of data for
    # 8 Hz at sfreq=250 → 85 samples; round up to be safe.
    # MNE formula: n_taps = int(round(0.34 * sfreq)) | always odd
    n_taps = int(round(0.34 * sfreq))
    if n_taps % 2 == 0:
        n_taps += 1
    return firwin(n_taps, [l_freq, h_freq], pass_zero=False, fs=sfreq)


def make_notch_sos(freq, sfreq, quality=30.0):
    """
    Create an IIR notch filter (biquad) as second-order sections.
    quality=30 is the MNE default for notch_filter.
    """
    b, a = iirnotch(freq, quality, fs=sfreq)
    from scipy.signal import tf2sos
    return tf2sos(b, a)


def preprocess_window(eeg_raw, sfreq, fir_b, notch_sos_list, n_need, n_total):
    """
    Apply the same preprocessing used in training to a raw EEG window.

    Parameters
    ----------
    eeg_raw        : np.ndarray (n_ch, n_samples_total)  — includes padding
    sfreq          : float
    fir_b          : np.ndarray — FIR bandpass coefficients
    notch_sos_list : list of np.ndarray — SOS for each notch filter
    n_need         : int — samples in the final window (without padding)
    n_total        : int — total captured samples (with padding)

    Returns
    -------
    window : np.ndarray (1, n_ch, n_need)  — ready for clf.predict()
             or None if there are not enough samples
    """
    n_samples = eeg_raw.shape[1]

    if n_samples < n_need:
        return None

    # Use the last n_total samples (or all samples if shorter)
    n_use = min(n_samples, n_total)
    eeg   = eeg_raw[:, -n_use:]

    # --- [1] Average reference ---
    # Subtract the mean across all channels at each timepoint,
    # exactly like raw.set_eeg_reference("average") in MNE.
    eeg = eeg - eeg.mean(axis=0, keepdims=True)

    # --- [2] Filtro FIR bandpass 8-30 Hz (mesmo que MNE firwin) ---
    from scipy.signal import filtfilt
    eeg = filtfilt(fir_b, [1.0], eeg, axis=1)

    # --- [3] Filtro notch (25 Hz e 50 Hz) ---
    for sos in notch_sos_list:
        eeg = sosfiltfilt(sos, eeg, axis=1)

    # --- [4] Discard padding — keep only the classification window ---
    eeg_window = eeg[:, -n_need:]

    return eeg_window[np.newaxis, :, :]   # (1, n_ch, n_times)


# ============================================================
# MODEL LOADING
# ============================================================

def load_models(session_path):
    paths = {
        "gating": os.path.join(session_path, "model_gating.pkl"),
        "axis": os.path.join(session_path, "model_axis.pkl"),
        "direction": os.path.join(session_path, "model_direction.pkl"),
    }

    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
                f"Models not found in {session_path}: {missing}\n"
                f"Run main.py first to train the model."
        )

    clf_gate = joblib.load(paths["gating"])
    clf_axis = joblib.load(paths["axis"])
    clf_dir  = joblib.load(paths["direction"])

    print(f"{GREEN}✓ Models loaded from {session_path}{RESET}")

    # Confirma que os pipelines têm os passos esperados
    for name, clf in [("gating", clf_gate), ("axis", clf_axis), ("direction", clf_dir)]:
        steps = [s for s, _ in clf.steps]
        print(f"  {name}: pipeline {steps}")

    return clf_gate, clf_axis, clf_dir


# ============================================================
# BOARD
# ============================================================

def start_board():

    params = BrainFlowInputParams()
    params.serial_port = CONFIG["serial_port"]

    board    = BoardShim(CONFIG["board_id"], params)
    board_id = CONFIG["board_id"]

    eeg_channels      = BoardShim.get_eeg_channels(board_id)
    timestamp_channel = BoardShim.get_timestamp_channel(board_id)
    sfreq             = BoardShim.get_sampling_rate(board_id)

    board.prepare_session()
    board.start_stream()
    time.sleep(2)

    print(f"{GREEN}✓ Board ligado | {len(eeg_channels)} canais | {sfreq} Hz{RESET}\n")

    return board, eeg_channels, timestamp_channel, sfreq


def stop_board(board):
    board.stop_stream()
    board.release_session()


# ============================================================
# ACQUIRE + PREPROCESS + CLASSIFY
# ============================================================

def get_window(board, eeg_channels, sfreq,
               fir_b, notch_sos_list,
               window_sec, pad_sec):
    """
    Wait window_sec + pad_sec seconds, apply training-identical preprocessing,
    and return (1, n_ch, n_times).

    The raw signal is in µV (BrainFlow returns µV).
    Training converts to V (x1e-6), and CSP/Scaler/LDA
    were trained on V — so we convert here too.
    """
    total_sec = window_sec + pad_sec
    n_need    = int(round(sfreq * window_sec))
    n_total   = int(round(sfreq * total_sec))

    # Clear the buffer before starting the timer
    board.get_board_data()

    time.sleep(total_sec)

    data = board.get_board_data()

    if data.shape[1] == 0:
        return None

    # BrainFlow devolve µV — converter para V como no treino
    eeg = data[eeg_channels, :] * 1e-6   # (n_ch, n_samples)

    return preprocess_window(eeg, sfreq, fir_b, notch_sos_list, n_need, n_total)


def classify(window, clf_gate, clf_axis, clf_dir):
    """
    Cascata de classificadores.
    O Pipeline (CSP → Scaler → LDA) faz tudo internamente.
    Devolve (pred_final, caminho).
    """
    # [1] GATING
    pred_gate = clf_gate.predict(window)[0]

    if pred_gate == 0:
        return 0, ["GATING → REST"]

    # [2] AXIS
    pred_axis = clf_axis.predict(window)[0]

    if pred_axis == 1:
        return 3, ["GATING → ACTIVE", "AXIS → FEET"]

    # [3] DIRECTION
    pred_dir = clf_dir.predict(window)[0]

    return int(pred_dir), ["GATING → ACTIVE", "AXIS → HANDS", f"DIRECTION → {NAMES[int(pred_dir)]}"]


# ============================================================
# TRIAL
# ============================================================

def run_trial(board, eeg_channels, sfreq,
              fir_b, notch_sos_list,
              clf_gate, clf_axis, clf_dir,
              target_label):
    """
    Run a trial: countdown → imagery → classification → validation.
    Returns (pred, correct).
    """
    # Conta decrescente
    for i in range(PREP_SEC, 0, -1):
        print(f"\r  {DIM}Starting in {i}...{RESET}  ", end="", flush=True)
        time.sleep(1.0)

    print(f"\r  {BOLD}{CYAN}PENSA: {SYMBOLS[target_label]} {NAMES[target_label]}{RESET}          ")
    print(f"  {DIM}(classifying {WINDOW_SEC:.0f}s of signal + {FILTER_PAD_SEC:.0f}s filter padding...){RESET}")

    # Adquire, pré-processa e classifica
    window = get_window(
        board, eeg_channels, sfreq,
        fir_b, notch_sos_list,
        WINDOW_SEC, FILTER_PAD_SEC
    )

    if window is None:
        print(f"  {RED}ERROR: insufficient EEG data.{RESET}")
        return None, None

    pred, path = classify(window, clf_gate, clf_axis, clf_dir)

    # Mostra resultado
    print()
    print(f"  {'─' * 40}")
    for step in path:
        print(f"  {DIM}{step}{RESET}")
    print(f"  {'─' * 40}")
    print(f"  Classification: {BOLD}{CYAN}{SYMBOLS[pred]} {NAMES[pred]}{RESET}")
    print(f"  {'─' * 40}")

    correct = (pred == target_label)

    if correct:
        print(f"  {GREEN}{BOLD}✓ CORRECT{RESET}")
    else:
        print(f"  {RED}{BOLD}✗ WRONG{RESET}  (was {NAMES[target_label]})")

    print()
    val = input(
        f"  Agree? [{GREEN}y{RESET}/{RED}n{RESET}/Enter=yes] "
    ).strip().lower()

    if val == "n":
        print(f"  Classes: {', '.join(f'{v}={k}' for k, v in NAMES.items())}")
        try:
            override = int(input("  True class (0/1/2/3): ").strip())
            correct  = (override == target_label)
            print(f"  {DIM}Corrected to {NAMES.get(override, str(override))}{RESET}")
        except ValueError:
            pass

    return pred, correct


# ============================================================
# METRICS
# ============================================================

def print_metrics(results):

    if not results:
        return

    targets = [r[0] for r in results]
    preds   = [r[1] for r in results]
    correct = [r[2] for r in results]

    n_total   = len(results)
    n_correct = sum(correct)
    accuracy  = n_correct / n_total

    print("\n" + "=" * 50)
    print(f"{BOLD}TEST SESSION RESULTS{RESET}")
    print("=" * 50)
    print(f"  Trials:   {n_total}")
    print(f"  Correct: {n_correct}")
    print(
        f"  Accuracy: "
        f"{GREEN if accuracy >= 0.7 else YELLOW if accuracy >= 0.5 else RED}"
        f"{BOLD}{accuracy*100:.1f}%{RESET}"
    )

    print(f"\n  {'Classe':<10} {'Trials':>6} {'Corretos':>8} {'Acc':>6}")
    print(f"  {'─'*34}")

    for label, name in NAMES.items():
        cls_trials  = [r for r in results if r[0] == label]
        cls_correct = sum(r[2] for r in cls_trials)
        cls_n       = len(cls_trials)

        if cls_n == 0:
            continue

        cls_acc = cls_correct / cls_n
        color   = GREEN if cls_acc >= 0.7 else YELLOW if cls_acc >= 0.5 else RED

        print(
            f"  {name:<10} {cls_n:>6} {cls_correct:>8} "
            f"{color}{cls_acc*100:>5.1f}%{RESET}"
        )

    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(targets, preds)
        print(f"\n  Cohen's κ: {BOLD}{kappa:.3f}{RESET}")
    except Exception:
        pass

    print("=" * 50 + "\n")


# ============================================================
# MAIN LOOP
# ============================================================

def main(session_path):

    print("\n" + "=" * 50)
    print(f"{BOLD}TESTE DO MODELO{RESET}")
    print(f"Sessão: {session_path}")
    print("=" * 50 + "\n")

    clf_gate, clf_axis, clf_dir = load_models(session_path)

    board, eeg_channels, timestamp_channel, sfreq = start_board()

    # Cria filtros uma vez — reutilizados em todos os trials
    fir_b          = make_fir_bandpass(L_FREQ, H_FREQ, sfreq)
    notch_sos_list = [make_notch_sos(f, sfreq) for f in NOTCH_FREQS]

    print(f"{DIM}Filtros criados: FIR bandpass {L_FREQ}-{H_FREQ} Hz + "
          f"notch {NOTCH_FREQS} Hz{RESET}\n")

    results = []
    classes = list(NAMES.keys())

    try:
        print(f"{BOLD}Comandos:{RESET}")
        print(f"  Enter    → next trial (random class)")
        print(f"  0/1/2/3  → escolhe a classe manualmente")
        print(f"  q        → quit and view metrics\n")

        trial_num = 0

        while True:

            trial_num += 1
            print(f"\n{'─' * 50}")
            print(f"  {DIM}Trial #{trial_num}{RESET}")

            cmd = input(
                f"  Classe? [{'/'.join(str(c) for c in classes)}/Enter=aleatório/q=sair] "
            ).strip().lower()

            if cmd == "q":
                break

            if cmd in [str(c) for c in classes]:
                target = int(cmd)
            else:
                target = int(np.random.choice(classes))

            print(f"\n  {BOLD}Prepare to think about: {SYMBOLS[target]} {NAMES[target]}{RESET}")

            pred, correct = run_trial(
                board, eeg_channels, sfreq,
                fir_b, notch_sos_list,
                clf_gate, clf_axis, clf_dir,
                target
            )

            if pred is not None:
                results.append((target, pred, correct))

    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}Interrupted.{RESET}")

    finally:
        stop_board(board)
        print_metrics(results)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        sessions = sorted([
            os.path.join("data", d)
            for d in os.listdir("data")
            if os.path.isdir(os.path.join("data", d))
            and os.path.exists(os.path.join("data", d, "model_gating.pkl"))
        ])

        if not sessions:
            print("No session with models found in data/")
            print("Usage: python test_model.py data/P001_20250521_143000")
            sys.exit(1)

        session_path = sessions[-1]
        print(f"Using most recent session: {session_path}")
    else:
        session_path = sys.argv[1]

    main(session_path)
# ============================================================
# FILE: evaluate_classifier.py
#
# Avalia o classificador numa sessão gerada, sem robô.
# Pré-processamento idêntico ao treino:
#   1. Referência average
#   2. Bandpass FIR 8-30Hz  (MNE firwin)
#   3. Notch FIR 50Hz       (MNE)
#   4. ICA — remoção automática de artefactos musculares
#
# Classificação blind: o classificador não consulta os markers
# para saber onde classificar — usa apenas a estrutura temporal
# fixa dos blocos (CUE_DURATION + EPOCH_TMIN/TMAX).
# Os markers são usados APENAS no final para verificar acertos.
#
# Estrutura de cada bloco (BLOCK_DURATION segundos):
#   [0s → CUE_DURATION]                        cue — ignorado
#   [CUE_DURATION+EPOCH_TMIN → CUE_DURATION+EPOCH_TMAX]  classificar
#   [CUE_DURATION+EPOCH_TMAX → BLOCK_DURATION] ignorado
#
# Uso:
#   python evaluate_classifier.py <session_path>
#   python evaluate_classifier.py data/S020_demo
# ============================================================

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import mne
from mne.preprocessing import ICA

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

try:
    from config import CONFIG
    _DEFAULT_SFREQ = CONFIG.get("sfreq", 160)
except Exception:
    _DEFAULT_SFREQ = 160

# ============================================================
# PARÂMETROS — têm de coincidir com o treino e com o generate
# ============================================================

# Janela de classificação (relativa a mi_start, i.e. após o cue)
EPOCH_TMIN  = 1.0
EPOCH_TMAX  = 3.5

# Estrutura temporal dos blocos — tem de coincidir com o generate
CUE_DURATION   = 2.0   # segundos de cue antes do mi_start
BLOCK_DURATION = 6.0   # CUE_DURATION + MI_DURATION (2.0 + 4.0)

# Janela absoluta dentro de cada bloco que o classificador usa:
#   bloco começa em t_block
#   mi_start está em t_block + CUE_DURATION
#   classificar de t_block + CUE_DURATION + EPOCH_TMIN
#              até t_block + CUE_DURATION + EPOCH_TMAX
CLASSIFY_ONSET  = CUE_DURATION + EPOCH_TMIN   # = 3.0s após início do bloco
CLASSIFY_OFFSET = CUE_DURATION + EPOCH_TMAX   # = 5.5s após início do bloco

L_FREQ      = 8.0
H_FREQ      = 30.0
NOTCH_FREQ  = 50.0
RANDOM_SEED = 42

NAMES       = {0: "REST", 1: "LEFT", 2: "RIGHT", 3: "FEET"}
LABELS      = [0, 1, 2, 3]

GREEN   = "\033[92m"
RED     = "\033[91m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"

CLASS_COLOR = {0: DIM, 1: CYAN, 2: MAGENTA, 3: YELLOW}


# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================

def preprocess_mne(eeg, sfreq, ch_names):
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw  = mne.io.RawArray(eeg, info, verbose=False)

    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore", verbose=False)

    raw.set_eeg_reference("average", verbose=False)
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=NOTCH_FREQ, method="fir", verbose=False)

    n_components = min(len(ch_names) - 1, 7)
    ica = ICA(n_components=n_components, random_state=RANDOM_SEED,
              method="fastica", max_iter=500)
    ica.fit(raw, verbose=False)
    muscle_idx, _ = ica.find_bads_muscle(raw, verbose=False)
    if muscle_idx:
        ica.exclude = muscle_idx
        ica.apply(raw, verbose=False)

    return raw


# ============================================================
# CARREGAR DADOS
# ============================================================

def load_session(session_path):
    for fname in ("eeg_raw.csv", "eeg_data.csv"):
        p = os.path.join(session_path, fname)
        if os.path.exists(p):
            eeg_path = p
            break
    else:
        raise FileNotFoundError(f"Nenhum ficheiro EEG em '{session_path}'")

    df      = pd.read_csv(eeg_path).sort_values("timestamp").reset_index(drop=True)
    ch_cols = [c for c in df.columns if c.startswith("ch_")]
    eeg     = df[ch_cols].values.T    # (n_ch, n_samples)
    times   = df["timestamp"].values

    markers = pd.read_csv(os.path.join(session_path, "markers.csv"))

    sfreq_path = os.path.join(session_path, "sfreq.txt")
    sfreq = float(open(sfreq_path).read()) if os.path.exists(sfreq_path) else float(_DEFAULT_SFREQ)

    return eeg, times, markers, sfreq, ch_cols

def load_models(session_path):
    keys    = ("gating", "axis", "direction")
    missing = [k for k in keys
               if not os.path.exists(os.path.join(session_path, f"model_{k}.pkl"))]
    if missing:
        raise FileNotFoundError(f"Modelos não encontrados: {missing}")
    return tuple(joblib.load(os.path.join(session_path, f"model_{k}.pkl")) for k in keys)


# ============================================================
# CLASSIFICAÇÃO
# Usa classes_ reais do modelo — robusto a qualquer mapeamento
# de IDs interno do MNE.
# ============================================================

def classify(epoch, clf_gate, clf_axis, clf_dir):
    """epoch: (n_ch, n_samples) → label 0–3"""
    w = epoch[np.newaxis, :, :]

    if clf_gate.predict(w)[0] == 0:
        return 0

    feet_label = clf_axis.classes_[1]
    if clf_axis.predict(w)[0] == feet_label:
        return 3

    pred_dir   = clf_dir.predict(w)[0]
    left_label = clf_dir.classes_[0]
    return 1 if pred_dir == left_label else 2


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

def print_confusion_matrix(y_true, y_pred):
    present = sorted(set(y_true) | set(y_pred))
    n       = len(present)
    idx     = {l: i for i, l in enumerate(present)}
    cm      = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t]][idx[p]] += 1

    col_w = 8
    lbl_w = 7

    print(f"\n  {'Matriz de Confusão':^{lbl_w + col_w * n}}")
    print(f"  {' '*lbl_w}{'predicted':^{col_w * n}}")
    print(f"  {' '*lbl_w}" + "".join(f"{NAMES[l]:>{col_w}}" for l in present))
    print(f"  {' '*lbl_w}{'─' * (col_w * n)}")

    for i, tl in enumerate(present):
        row = f"  {NAMES[tl]:>{lbl_w-1}} │"
        for j in range(n):
            v = cm[i][j]
            if i == j:
                row += f"{GREEN}{v:>{col_w}}{RESET}"
            elif v > 0:
                row += f"{RED}{v:>{col_w}}{RESET}"
            else:
                row += f"{DIM}{v:>{col_w}}{RESET}"
        row += f"  │ {NAMES[tl]}"
        print(row)

    print(f"  {' '*lbl_w}{'─' * (col_w * n)}")
    print(f"\n  {'Classe':<10} {'Corretas':>9} {'Total':>7} {'Acurácia':>10}")
    print(f"  {'─'*42}")
    for i, label in enumerate(present):
        total   = cm[i, :].sum()
        correct = cm[i, i]
        acc     = correct / total * 100 if total > 0 else 0.0
        color   = CLASS_COLOR.get(label, RESET)
        bar     = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"  {color}{NAMES[label]:<10}{RESET} {correct:>9} {total:>7} "
              f"   {BOLD}{acc:>5.1f}%{RESET}  {bar}")


# ============================================================
# MAIN
# ============================================================

def run(session_path):
    print(f"\n{BOLD}{'═'*62}{RESET}")
    print(f"{BOLD}  EVALUATE CLASSIFIER  (blind timing){RESET}")
    print(f"  {DIM}{session_path}{RESET}")
    print(f"  bloco: {BLOCK_DURATION}s  |  cue: {CUE_DURATION}s  |  "
          f"época: [{EPOCH_TMIN}s → {EPOCH_TMAX}s] após mi_start")
    print(f"  → classifica [{CLASSIFY_ONSET}s → {CLASSIFY_OFFSET}s] em cada bloco")
    print(f"  preprocessing: avg ref → bandpass {L_FREQ}–{H_FREQ}Hz → "
          f"notch {NOTCH_FREQ}Hz → ICA")
    print(f"{BOLD}{'═'*62}{RESET}\n")

    # Modelos
    clf_gate, clf_axis, clf_dir = load_models(session_path)
    print(f"{GREEN}✓ Modelos carregados{RESET}")
    print(f"  gate  classes: {clf_gate.classes_}")
    print(f"  axis  classes: {clf_axis.classes_}  "
          f"(Mãos={clf_axis.classes_[0]}, Pés={clf_axis.classes_[1]})")
    print(f"  dir   classes: {clf_dir.classes_}  "
          f"(LEFT={clf_dir.classes_[0]}, RIGHT={clf_dir.classes_[1]})")

    # Dados
    eeg, times, markers, sfreq, ch_cols = load_session(session_path)
    total_duration = times[-1]
    print(f"{GREEN}✓ EEG: {eeg.shape[0]} canais × {eeg.shape[1]} amostras "
          f"@ {sfreq:.0f}Hz  ({total_duration:.1f}s){RESET}")

    # Pré-processamento
    print(f"{DIM}A pré-processar EEG...{RESET}", end="", flush=True)
    raw      = preprocess_mne(eeg, sfreq, ch_cols)
    eeg_proc = raw.get_data()
    print(f"\r{GREEN}✓ EEG pré-processado{RESET}                        \n")

    # Ground truth a partir dos markers (só para verificação final)
    mi_markers = (markers[markers["event"] == "mi_start"]
                  .sort_values("timestamp")
                  .reset_index(drop=True))

    # ----------------------------------------------------------
    # Classificação blind: avança bloco a bloco pela timeline
    # sem consultar os markers
    # ----------------------------------------------------------
    n_blocks    = int(total_duration / BLOCK_DURATION)
    block_times = [b * BLOCK_DURATION for b in range(n_blocks)]

    print(f"  {n_blocks} blocos × {BLOCK_DURATION}s  "
          f"| classifica [{CLASSIFY_ONSET}s → {CLASSIFY_OFFSET}s] em cada bloco\n")
    print(f"  {'Bloco':>5}  {'t_class':>8}  {'Predição':<10}  "
          f"{'Ground Truth':<12}  {'':6}  {'Nota'}")
    print(f"  {'─'*62}")

    predictions = []   # (t_block, pred)

    for b, t_block in enumerate(block_times):
        t_s = t_block + CLASSIFY_ONSET
        t_e = t_block + CLASSIFY_OFFSET

        mask = (times >= t_s) & (times < t_e)
        if mask.sum() < 2:
            print(f"  {b+1:>5}  {t_s:>7.1f}s  {DIM}sem amostras — saltado{RESET}")
            continue

        epoch = eeg_proc[:, mask]
        pred  = classify(epoch, clf_gate, clf_axis, clf_dir)
        predictions.append((t_block, pred))

        pc = CLASS_COLOR.get(pred, RESET)
        pred_s = f"{pc}{BOLD}{NAMES[pred]:<10}{RESET}"

        # Procura o marker correspondente a este bloco (para display)
        t_mi_expected = t_block + CUE_DURATION
        match = mi_markers[np.abs(mi_markers["timestamp"] - t_mi_expected) < 0.5]
        if not match.empty:
            gt    = int(match.iloc[0]["label"])
            gc    = CLASS_COLOR.get(gt, RESET)
            gt_s  = f"{gc}{BOLD}{NAMES[gt]:<12}{RESET}"
            if pred == gt:
                icon, note = f"{GREEN}✓{RESET}", ""
            elif pred == 0 or gt == 0:
                icon, note = f"{YELLOW}~{RESET}", f"{DIM}REST mismatch{RESET}"
            else:
                icon, note = f"{RED}✗{RESET}", f"{DIM}esperado {NAMES[gt]}{RESET}"
        else:
            gt_s  = f"{DIM}{'?':<12}{RESET}"
            icon, note = " ", f"{DIM}sem marker{RESET}"

        print(f"  {b+1:>5}  {t_s:>7.1f}s  {pred_s}  {gt_s}  {icon}      {note}")

    # ----------------------------------------------------------
    # Verificação: associa cada predição ao marker mais próximo
    # ----------------------------------------------------------
    y_true = []
    y_pred = []

    for t_block, pred in predictions:
        t_mi_expected = t_block + CUE_DURATION
        match = mi_markers[np.abs(mi_markers["timestamp"] - t_mi_expected) < 0.5]
        if match.empty:
            continue
        gt = int(match.iloc[0]["label"])
        y_true.append(gt)
        y_pred.append(pred)

    if not y_true:
        print(f"\n{RED}Nenhuma época classificada.{RESET}")
        return

    y_true    = np.array(y_true)
    y_pred    = np.array(y_pred)
    n_total   = len(y_true)
    n_correct = (y_true == y_pred).sum()
    acc       = n_correct / n_total * 100

    print(f"\n{BOLD}{'═'*62}{RESET}")
    print(f"{BOLD}  RESULTADOS{RESET}")
    print(f"{'─'*62}")
    print(f"  Blocos classificados : {n_total}")
    print(f"  Correctos            : {GREEN}{n_correct}{RESET}")
    print(f"  Errados              : {RED}{n_total - n_correct}{RESET}")
    print(f"  Acurácia global      : {BOLD}{acc:.1f}%{RESET}")

    print_confusion_matrix(y_true, y_pred)
    print(f"\n{BOLD}{'═'*62}{RESET}\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Avalia o classificador BCI com timing blind (sem consultar markers)."
    )
    parser.add_argument("session_path", nargs="?",
                        help="Pasta da sessão (eeg_raw.csv, markers.csv, modelos .pkl)")
    args = parser.parse_args()

    if args.session_path:
        session_path = args.session_path
    else:
        data_dir = "data"
        sessions = []
        if os.path.isdir(data_dir):
            sessions = sorted([
                os.path.join(data_dir, d)
                for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
                and os.path.exists(os.path.join(data_dir, d, "model_gating.pkl"))
                and any(os.path.exists(os.path.join(data_dir, d, f))
                        for f in ("eeg_raw.csv", "eeg_data.csv"))
            ])
        if not sessions:
            print("Uso: python evaluate_classifier.py <session_path>")
            sys.exit(1)
        session_path = sessions[-1]
        print(f"A usar sessão mais recente: {session_path}")

    run(session_path)
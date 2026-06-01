# ============================================================
# FILE: playback_robot.py
#
# Controlo do robô em modo playback guiado por eventos.
#
# Protocolo (igual ao que seria em real-time):
#   cue_on   → mostra countdown + classe alvo
#   mi_start → aguarda EPOCH_TMIN (1s de settling)
#              extrai [mi_start + EPOCH_TMIN, mi_start + EPOCH_TMAX]
#              classifica → envia comando ao robô
#   mi_end   → mostra resultado, aguarda próximo cue
#
# Pré-processamento idêntico ao treino (MNE) — igual ao evaluate_classifier:
#   1. Referência average
#   2. Bandpass FIR 8–30 Hz  (MNE firwin)
#   3. Notch FIR 50 Hz       (MNE)
#   4. ICA — remoção automática de artefactos musculares
#
# Uso:
#   python playback_robot.py <session_path> [serial_port] [--no-robot] [--speed N]
#   python playback_robot.py data/S020_demo --no-robot
#   python playback_robot.py data/S020_demo COM6
#   python playback_robot.py data/S020_demo COM6 --speed 2.0
# ============================================================

import os
import sys
import time
import joblib
import collections
import argparse
import warnings
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    from config import CONFIG
    _DEFAULT_SFREQ = CONFIG.get("sfreq", 160)
except Exception:
    _DEFAULT_SFREQ = 160


# ============================================================
# PARÂMETROS — têm de coincidir com o treino e com o evaluate
# ============================================================

# Janela de MI relativa ao mi_start — igual ao treino
EPOCH_TMIN = 1.0    # segundos após mi_start (settling)
EPOCH_TMAX = 3.5    # segundos após mi_start

# Filtro — igual ao treino e ao evaluate_classifier
L_FREQ     = 8.0
H_FREQ     = 30.0
NOTCH_FREQ = 50.0
RANDOM_SEED = 42

ROBOT_BAUD     = 9600
PLAYBACK_SPEED = 1.0   # 1.0 = tempo real | 2.0 = 2× | 0 = máximo

COMMAND_MAP = {0: None, 1: "LEFT", 2: "RIGHT", 3: "FEET"}
NAMES       = {0: "REST", 1: "LEFT", 2: "RIGHT", 3: "FEET"}
SYMBOLS     = {0: "■",   1: "←",   2: "→",     3: "↓"}

# Servo activo por defeito (para mostrar no terminal)
# O robô cicla entre servos com o comando FEET
SERVO_NAMES = {0: "Servo 0 (pinça)", 1: "Servo 1 (cotovelo)",
               2: "Servo 2 (ombro)", 3: "Servo 3 (base)"}

# Cores ANSI
GREEN   = "\033[92m"
RED     = "\033[91m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"

CLASS_COLOR = {0: DIM, 1: CYAN, 2: MAGENTA, 3: YELLOW}
CLASS_BG    = {0: "\033[100m", 1: "\033[44m", 2: "\033[45m", 3: "\033[43m\033[30m"}


# ============================================================
# PRÉ-PROCESSAMENTO MNE — idêntico ao evaluate_classifier
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
        raise FileNotFoundError(
            f"Nenhum ficheiro EEG em '{session_path}' (eeg_raw.csv / eeg_data.csv)")

    df      = pd.read_csv(eeg_path).sort_values("timestamp").reset_index(drop=True)
    ch_cols = [c for c in df.columns if c.startswith("ch_")]
    if not ch_cols:
        raise ValueError("Nenhuma coluna ch_* no CSV de EEG")

    eeg   = df[ch_cols].values.T    # (n_ch, n_samples)
    times = df["timestamp"].values  # (n_samples,)

    markers_path = os.path.join(session_path, "markers.csv")
    if not os.path.exists(markers_path):
        raise FileNotFoundError(f"markers.csv não encontrado em '{session_path}'")
    markers = pd.read_csv(markers_path)

    sfreq_path = os.path.join(session_path, "sfreq.txt")
    if os.path.exists(sfreq_path):
        sfreq = float(open(sfreq_path).read().strip())
    else:
        dt    = np.diff(times[:min(500, len(times))])
        sfreq = float(1.0 / np.median(dt[dt > 0]))

    return eeg, times, markers, sfreq, ch_cols

def load_models(session_path):
    paths = {k: os.path.join(session_path, f"model_{k}.pkl")
             for k in ("gating", "axis", "direction")}
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Modelos não encontrados: {missing}")
    return (joblib.load(paths["gating"]),
            joblib.load(paths["axis"]),
            joblib.load(paths["direction"]))


# ============================================================
# EXTRAIR ÉPOCA (pelo timestamp, sem olhar para o label)
# ============================================================

def extract_epoch(eeg_proc, times, t_start, t_end):
    """
    Extrai amostras de eeg_proc entre t_start e t_end.
    Devolve (n_ch, n_samples) ou None se fora dos limites.
    """
    mask = (times >= t_start) & (times <= t_end)
    if mask.sum() < 2:
        return None
    return eeg_proc[:, mask]


# ============================================================
# CLASSIFICAÇÃO — idêntica ao evaluate_classifier
# Usa classes_ reais do modelo — robusto a qualquer mapeamento
# de IDs interno do MNE.
# ============================================================

def classify(epoch, clf_gate, clf_axis, clf_dir):
    """epoch: (n_ch, n_samples) → label 0–3"""
    w = epoch[np.newaxis, :, :]           # (1, n_ch, n_samples)

    if clf_gate.predict(w)[0] == 0:
        return 0

    feet_label = clf_axis.classes_[1]
    if clf_axis.predict(w)[0] == feet_label:
        return 3

    pred_dir   = clf_dir.predict(w)[0]
    left_label = clf_dir.classes_[0]
    return 1 if pred_dir == left_label else 2


# ============================================================
# ROBÔ
# ============================================================

def open_robot(port):
    if not HAS_SERIAL:
        raise RuntimeError("pyserial não instalado: pip install pyserial")
    rob = serial.Serial(port, ROBOT_BAUD, timeout=1)
    time.sleep(2)
    if rob.in_waiting:
        print(f"  Arduino: {rob.readline().decode(errors='replace').strip()}")
    return rob

def send_command(rob, cmd):
    if not rob or not cmd:
        return
    try:
        rob.write((cmd + "\n").encode())
        time.sleep(0.05)
        if rob.in_waiting:
            print(f"  {DIM}Arduino → {rob.readline().decode(errors='replace').strip()}{RESET}")
    except Exception as e:
        print(f"  {RED}Erro serial: {e}{RESET}")

def close_robot(rob):
    if rob and rob.is_open:
        rob.close()


# ============================================================
# OUTPUT NO TERMINAL
# ============================================================

def divider(char="─", n=58):
    print(f"  {char * n}")

def cls():
    os.system("cls" if os.name == "nt" else "clear")

def print_header(session_path, robot_connected, sfreq, total_events, ev_idx):
    cls()
    print(f"{BOLD}{'═' * 62}{RESET}")
    print(f"{BOLD}  INSERT-BRAIN  ·  PLAYBACK BCI{RESET}")
    print(f"  {DIM}{session_path}{RESET}")
    rob_s = f"{GREEN}● robô ligado{RESET}" if robot_connected else f"{DIM}○ sem robô{RESET}"
    print(f"  {rob_s}   {DIM}{sfreq:.0f}Hz · época [{EPOCH_TMIN}–{EPOCH_TMAX}s] · "
          f"avg ref → FIR {L_FREQ}–{H_FREQ}Hz → notch {NOTCH_FREQ}Hz → ICA{RESET}")
    print(f"  Evento {ev_idx} / {total_events}")
    print(f"{BOLD}{'═' * 62}{RESET}\n")

def print_cue(gt_label, countdown_sec, speed):
    color = CLASS_COLOR.get(gt_label, RESET)
    bg    = CLASS_BG.get(gt_label, "")
    name  = NAMES.get(gt_label, "?")
    sym   = SYMBOLS.get(gt_label, "?")
    print(f"  {BOLD}CUE{RESET}  →  {bg}{BOLD}  {sym} {name}  {RESET}\n")

    if speed > 0 and countdown_sec > 0:
        steps = int(countdown_sec)
        for i in range(steps, 0, -1):
            print(f"\r  {DIM}Aguarda...  {BOLD}{i}{RESET}  ", end="", flush=True)
            time.sleep(1.0 / speed)
        print(f"\r  {GREEN}{BOLD}CLASSIFICA!{RESET}            ")

def print_result(ev_idx, gt_label, pred, cmd, servo_idx,
                 n_correct, n_wrong, history):
    color_gt   = CLASS_COLOR.get(gt_label, RESET)
    color_pred = CLASS_COLOR.get(pred, RESET)

    gt_str   = f"{color_gt}{BOLD}{SYMBOLS[gt_label]} {NAMES[gt_label]}{RESET}"
    pred_str = f"{color_pred}{BOLD}{SYMBOLS[pred]} {NAMES[pred]}{RESET}"

    if pred == gt_label:
        icon = f"{GREEN}✓  CORRECTO{RESET}"
    elif pred == 0 or gt_label == 0:
        icon = f"{YELLOW}~  REST/ACTIVE mismatch{RESET}"
    else:
        icon = f"{RED}✗  ERRADO{RESET}"

    divider()
    print(f"  Ground truth  :  {gt_str}")
    print(f"  Classificação :  {pred_str}   {icon}")

    if cmd:
        servo_s = SERVO_NAMES.get(servo_idx, f"Servo {servo_idx}")
        print(f"  Comando robô  :  {GREEN}{BOLD}{cmd}{RESET}   {DIM}({servo_s}){RESET}")
    else:
        print(f"  Comando robô  :  {DIM}nenhum (REST){RESET}")

    total = n_correct + n_wrong
    acc   = n_correct / total * 100 if total > 0 else 0.0
    divider()
    print(f"  Acurácia acumulada: {BOLD}{acc:.1f}%{RESET}  "
          f"({GREEN}✓ {n_correct}{RESET}  {RED}✗ {n_wrong}{RESET}  de {total})\n")

    if history:
        print(f"  {DIM}Histórico:{RESET}")
        for e in history[-8:]:
            gc = CLASS_COLOR.get(e["gt"],   RESET)
            pc = CLASS_COLOR.get(e["pred"], RESET)
            ic = (f"{GREEN}✓{RESET}" if e["pred"] == e["gt"]
                  else f"{YELLOW}~{RESET}" if (e["pred"] == 0 or e["gt"] == 0)
                  else f"{RED}✗{RESET}")
            cs = f"{GREEN}{e['cmd']}{RESET}" if e["cmd"] else f"{DIM}—{RESET}"
            print(f"    [{e['idx']:>2}]  "
                  f"gt:{gc}{NAMES[e['gt']]:<5}{RESET}  "
                  f"pred:{pc}{NAMES[e['pred']]:<5}{RESET}  "
                  f"{ic}  {cs}")
    print()


# ============================================================
# LOOP PRINCIPAL
# ============================================================

def run(session_path, robot_port=None, no_robot=False):

    print(f"\n{BOLD}A carregar sessão...{RESET}")

    clf_gate, clf_axis, clf_dir = load_models(session_path)
    print(f"{GREEN}✓ Modelos carregados{RESET}")
    print(f"  gate  classes: {clf_gate.classes_}")
    print(f"  axis  classes: {clf_axis.classes_}  "
          f"(Mãos={clf_axis.classes_[0]}, Pés={clf_axis.classes_[1]})")
    print(f"  dir   classes: {clf_dir.classes_}  "
          f"(LEFT={clf_dir.classes_[0]}, RIGHT={clf_dir.classes_[1]})")

    eeg, times, markers, sfreq, ch_cols = load_session(session_path)
    print(f"{GREEN}✓ EEG: {eeg.shape[0]} canais × {eeg.shape[1]} amostras "
          f"@ {sfreq:.1f}Hz  ({times[-1]:.1f}s){RESET}")
    print(f"{GREEN}✓ Markers: {len(markers)} linhas{RESET}")

    # Pré-processamento MNE — idêntico ao evaluate_classifier
    print(f"{DIM}A pré-processar EEG (avg ref → bandpass → notch → ICA)...{RESET}",
          end="", flush=True)
    raw      = preprocess_mne(eeg, sfreq, ch_cols)
    eeg_proc = raw.get_data()
    print(f"\r{GREEN}✓ EEG pré-processado (avg ref → FIR {L_FREQ}–{H_FREQ}Hz → "
          f"notch {NOTCH_FREQ}Hz → ICA){RESET}")

    # Eventos relevantes: pares (cue_on, mi_start, mi_end, label)
    cues   = markers[markers["event"] == "cue_on"].sort_values("timestamp").reset_index(drop=True)
    starts = markers[markers["event"] == "mi_start"].sort_values("timestamp").reset_index(drop=True)
    ends   = markers[markers["event"] == "mi_end"].sort_values("timestamp").reset_index(drop=True)

    # Alinha os três por índice (devem ter o mesmo número de linhas)
    n_events = min(len(cues), len(starts), len(ends))
    if n_events == 0:
        raise ValueError("Nenhum par cue_on/mi_start/mi_end encontrado nos markers.")
    print(f"{GREEN}✓ {n_events} épocas de MI identificadas{RESET}")

    # Robô
    rob = None
    robot_connected = False
    if not no_robot and robot_port:
        try:
            rob = open_robot(robot_port)
            robot_connected = True
            print(f"{GREEN}✓ Robô ligado em {robot_port}{RESET}")
        except Exception as e:
            print(f"{YELLOW}⚠ Robô não disponível: {e}{RESET}")

    # Estado do servo activo (FEET cicla 0→1→2→3→0)
    servo_idx = 0

    # Estatísticas
    stats     = collections.defaultdict(int)
    n_correct = 0
    n_wrong   = 0
    history   = []

    time.sleep(1.0)

    try:
        for i in range(n_events):

            t_cue      = float(cues.iloc[i]["timestamp"])
            t_mi_start = float(starts.iloc[i]["timestamp"])
            t_mi_end   = float(ends.iloc[i]["timestamp"])
            gt_label   = int(starts.iloc[i]["label"])

            # ── Header ────────────────────────────────────────────
            print_header(session_path, robot_connected, sfreq, n_events, i + 1)

            # ── CUE + countdown ───────────────────────────────────
            # Tempo de cue = t_mi_start - t_cue (tipicamente 2s)
            cue_duration = t_mi_start - t_cue
            print_cue(gt_label, cue_duration, PLAYBACK_SPEED)

            # ── Settling (EPOCH_TMIN após mi_start) ───────────────
            if PLAYBACK_SPEED > 0:
                time.sleep(EPOCH_TMIN / PLAYBACK_SPEED)

            # ── Extrai época e classifica ─────────────────────────
            t_win_start = t_mi_start + EPOCH_TMIN
            t_win_end   = t_mi_start + EPOCH_TMAX

            epoch = extract_epoch(eeg_proc, times, t_win_start, t_win_end)

            if epoch is None or epoch.shape[1] < 2:
                print(f"  {YELLOW}⚠ Época fora dos limites do EEG — a saltar{RESET}")
                time.sleep(1.0)
                continue

            pred = classify(epoch, clf_gate, clf_axis, clf_dir)
            cmd  = COMMAND_MAP.get(pred)
            stats[pred] += 1

            # Acurácia
            if pred == gt_label:
                n_correct += 1
            else:
                n_wrong += 1

            # Comando ao robô
            if cmd:
                send_command(rob, cmd)
                if pred == 3:   # FEET cicla o servo activo
                    servo_idx = (servo_idx + 1) % 4

            history.append({
                "idx":  i + 1,
                "gt":   gt_label,
                "pred": pred,
                "cmd":  cmd,
            })

            # ── Mostra resultado ──────────────────────────────────
            print_result(i + 1, gt_label, pred, cmd, servo_idx,
                         n_correct, n_wrong, history)

            # ── Aguarda até ao mi_end antes de avançar ────────────
            # (tempo que resta do MI após a classificação)
            remaining = (t_mi_end - t_win_end)
            if PLAYBACK_SPEED > 0 and remaining > 0:
                time.sleep(remaining / PLAYBACK_SPEED)

        print(f"\n{GREEN}✓ Playback concluído.{RESET}\n")

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Parado pelo utilizador.{RESET}\n")

    finally:
        close_robot(rob)

        total = n_correct + n_wrong
        acc   = n_correct / total * 100 if total > 0 else 0.0

        print("=" * 62)
        print(f"{BOLD}SUMÁRIO FINAL{RESET}")
        print(f"  Épocas classificadas : {total}")
        print(f"  Acurácia global      : {BOLD}{acc:.1f}%{RESET}  "
              f"({GREEN}✓ {n_correct}{RESET}  {RED}✗ {n_wrong}{RESET})")
        print(f"\n  {'Classe':<10} {'Predições':>10} {'Corretas':>10}")
        print(f"  {'─' * 34}")
        for label in [1, 2, 3, 0]:
            n_p  = stats[label]
            n_ok = sum(1 for e in history if e["pred"] == label and e["gt"] == label)
            col  = CLASS_COLOR.get(label, RESET)
            acc_ = f"({n_ok/n_p*100:.0f}%)" if n_p > 0 else ""
            print(f"  {col}{NAMES[label]:<10}{RESET} {n_p:>10}  {n_ok:>5} {DIM}{acc_}{RESET}")
        print("=" * 62 + "\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Playback BCI guiado por eventos — classifica épocas de MI e controla robô."
    )
    parser.add_argument("session_path", nargs="?",
                        help="Pasta da sessão (eeg_raw.csv / eeg_data.csv, markers.csv, modelos .pkl)")
    parser.add_argument("serial_port",  nargs="?", default=None,
                        help="Porta série do robô (ex: COM6, /dev/ttyUSB0)")
    parser.add_argument("--no-robot",   action="store_true", help="Corre sem robô")
    parser.add_argument("--speed",      type=float, default=1.0,
                        help="Velocidade de playback (1.0=tempo real, 2.0=2×, 0=máximo). Default: 1.0")
    parser.add_argument("--tmin",       type=float, default=EPOCH_TMIN,
                        help=f"Início da época após mi_start em segundos (default: {EPOCH_TMIN})")
    parser.add_argument("--tmax",       type=float, default=EPOCH_TMAX,
                        help=f"Fim da época após mi_start em segundos (default: {EPOCH_TMAX})")
    args = parser.parse_args()

    PLAYBACK_SPEED = args.speed
    EPOCH_TMIN     = args.tmin
    EPOCH_TMAX     = args.tmax

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
            print("Nenhuma sessão encontrada. Uso:")
            print("  python playback_robot.py <session_path> [serial_port] [--no-robot]")
            sys.exit(1)
        session_path = sessions[-1]
        print(f"A usar sessão mais recente: {session_path}")

    robot_port = args.serial_port
    if not args.no_robot and not robot_port:
        resp = input("Porta série do robô (Enter para correr sem robô): ").strip()
        robot_port = resp if resp else None

    run(session_path, robot_port=robot_port, no_robot=args.no_robot)
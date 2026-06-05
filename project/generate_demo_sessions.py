# ============================================================
# FILE: generate_demo_sessions.py
#
# Builds synthetic EEG sessions from PhysioNet EEGBCI data.
#
# Each session is composed of fixed-duration blocks:
#
#   [0s → CUE_DURATION]   cue / idle period (ignored by model)
#   [CUE_DURATION → end]  motor imagery segment (active class)
#
# Blocks are concatenated sequentially to simulate a real-time
# continuous EEG stream without relying on event markers during
# inference.
#
# Class mapping (PhysioNet → project labels):
#   T1 (hands) → LEFT
#   T2 (hands) → RIGHT
#   T2 (feet)  → FEET
#   baseline   → REST
#
# Output structure:
#   generated_sessions/S{subj}_demo/
#       eeg_raw.csv   -> EEG channels + timestamps
#       markers.csv   -> event timeline (cue / MI phases)
#       sfreq.txt     -> sampling frequency
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

# ============================================================
# CONFIG
# ============================================================

SUBJECTS = [20, 29, 48]
OUT_DIR  = "generated_sessions"

random.seed(42)
np.random.seed(42)

CHANNELS_8 = ["FC3", "FC4", "C3", "C4", "CP3", "CP4", "Cz", "FCz"]

RUNS_HANDS = [4, 8, 12]
RUNS_FEET  = [6, 10, 14]
RUNS_REST  = [1, 2]

N_COMMANDS   = 15     # nº de comandos activos na sessão

# Timing fixo por bloco — TODOS os blocos têm exactamente isto
CUE_DURATION   = 2.0  # segundos de cue (classificador ignora)
MI_DURATION    = 5.0  # segundos de MI real contínuo
BLOCK_DURATION = CUE_DURATION + MI_DURATION   # = 7.0s por bloco

# Janela que o classificador vai usar (tem de caber dentro de MI_DURATION)
EPOCH_TMIN = 1.0   # após mi_start
EPOCH_TMAX = 3.5   # após mi_start  →  3.5s < 5.0s ✓

assert EPOCH_TMAX <= MI_DURATION, "EPOCH_TMAX tem de caber dentro do MI!"

# ============================================================
# EXTRACÇÃO DE SEGMENTOS
# ============================================================

def extract_mi_segments(raw, min_duration_sec):
    """
    Extrai segmentos de EEG por classe (T0/T1/T2) a partir dos eventos.
    Só guarda segmentos com duração >= min_duration_sec.
    Devolve: dict {event_label: [array(n_ch, n_samp), ...]}
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    sfreq     = raw.info["sfreq"]
    min_samps = int(min_duration_sec * sfreq)
    data      = raw.get_data()
    reverse   = {v: k for k, v in event_id.items()}
    segs      = {}

    for i, ev in enumerate(events):
        label = reverse[ev[2]]
        start = ev[0]
        end   = events[i + 1, 0] if i + 1 < len(events) else data.shape[1]
        if (end - start) < min_samps:
            continue
        segs.setdefault(label, []).append(data[:, start : end])

    return segs


def extract_rest_segments(raw, min_duration_sec):
    """
    Extrai janelas deslizantes de baseline dos runs de olhos fechados.
    O sinal inteiro é REST — não há eventos MI.
    """
    sfreq     = raw.info["sfreq"]
    min_samps = int(min_duration_sec * sfreq)
    data      = raw.get_data()
    segs      = []
    for start in range(0, data.shape[1] - min_samps, min_samps):
        segs.append(data[:, start : start + min_samps])
    return segs


def pick_segment(seg, n_samps):
    """
    Pega num segmento e devolve exactamente n_samps amostras.
    Se o segmento for maior: escolhe um offset aleatório dentro dele.
    Se for mais curto: não devia acontecer (filtramos por min_duration).
    """
    if seg.shape[1] >= n_samps:
        max_offset = seg.shape[1] - n_samps
        offset = random.randint(0, max_offset)
        return seg[:, offset : offset + n_samps]
    else:
        # Fallback de segurança: repete o sinal
        reps = int(np.ceil(n_samps / seg.shape[1]))
        return np.tile(seg, (1, reps))[:, :n_samps]


# ============================================================
# BUILD SESSION
# ============================================================

def build_session(subject):
    print(f"\n{'='*54}")
    print(f"  Subject {subject}")
    print(f"{'='*54}")

    # --- Carrega e prepara os raws ---
    def load_runs(runs):
        raws = []
        for run in runs:
            p = eegbci.load_data(subject, runs=[run])[0]
            r = read_raw_edf(p, preload=True, verbose=False)
            eegbci.standardize(r)
            r.pick_channels(CHANNELS_8)
            raws.append(r)
        return raws

    print("  A carregar dados do PhysioNet...")
    raws_hands = load_runs(RUNS_HANDS)
    raws_feet  = load_runs(RUNS_FEET)
    raws_rest  = load_runs(RUNS_REST)

    sfreq = raws_hands[0].info["sfreq"]
    print(f"  sfreq: {sfreq} Hz")

    n_mi_samps   = int(MI_DURATION   * sfreq)   # amostras de MI por bloco
    n_blk_samps  = int(BLOCK_DURATION * sfreq)   # amostras totais por bloco

    # --- Extrai segmentos (precisam ter pelo menos MI_DURATION de duração) ---
    left_segs  = []
    right_segs = []
    feet_segs  = []

    for r in raws_hands:
        segs = extract_mi_segments(r, MI_DURATION)
        left_segs.extend(segs.get("T1", []))
        right_segs.extend(segs.get("T2", []))

    for r in raws_feet:
        segs = extract_mi_segments(r, MI_DURATION)
        feet_segs.extend(segs.get("T2", []))

    rest_segs = []
    for r in raws_rest:
        rest_segs.extend(extract_rest_segments(r, MI_DURATION))

    print(f"  Segmentos disponíveis:")
    print(f"    LEFT  (T1 hands): {len(left_segs)}")
    print(f"    RIGHT (T2 hands): {len(right_segs)}")
    print(f"    FEET  (T2 feet):  {len(feet_segs)}")
    print(f"    REST  (baseline): {len(rest_segs)}")

    for name, lst in [("LEFT", left_segs), ("RIGHT", right_segs),
                      ("FEET", feet_segs),  ("REST",  rest_segs)]:
        if not lst:
            raise ValueError(f"Sem segmentos para {name} no sujeito {subject}!")

    # --- Gera sequência de comandos ---
    # Garante que as 3 classes activas aparecem pelo menos N_COMMANDS//3 vezes
    active = ([1, 2, 3] * (N_COMMANDS // 3 + 1))[:N_COMMANDS]
    random.shuffle(active)

    # Intercala REST entre cada comando activo
    sequence = []
    for cls in active:
        sequence.append(0)    # REST
        sequence.append(cls)  # comando activo

    # --- Constrói EEG bloco a bloco ---
    eeg_blocks = []
    markers    = []
    t          = 0.0

    for label in sequence:

        # Escolhe e recorta o segmento de MI (exactamente n_mi_samps)
        pool = {0: rest_segs, 1: left_segs, 2: right_segs, 3: feet_segs}[label]
        mi_seg = pick_segment(random.choice(pool), n_mi_samps)

        # Constrói o bloco completo de n_blk_samps:
        #   [CUE: n_cue_samps de REST qualquer] + [MI: n_mi_samps da classe]
        # A descontinuidade fica no corte entre o bloco anterior e o cue —
        # o classificador nunca toca nesse período.
        n_cue_samps = n_blk_samps - n_mi_samps   # = CUE_DURATION * sfreq
        cue_seg = pick_segment(random.choice(rest_segs), n_cue_samps)

        block = np.concatenate([cue_seg, mi_seg], axis=1)  # (8, n_blk_samps)
        assert block.shape[1] == n_blk_samps

        # Markers alinhados com o EEG acumulado
        markers.append([t,                    "cue_on",   label])
        markers.append([t + CUE_DURATION,     "mi_start", label])
        markers.append([t + BLOCK_DURATION,   "mi_end",   label])

        eeg_blocks.append(block)
        t += BLOCK_DURATION

    # --- Concatena e exporta ---
    eeg       = np.concatenate(eeg_blocks, axis=1)   # (8, n_total)
    n_samples = eeg.shape[1]
    times     = np.arange(n_samples) / sfreq

    print(f"\n  EEG exportado:")
    print(f"    {n_samples} amostras = {n_samples/sfreq:.1f}s")
    print(f"    {len(sequence)} blocos × {BLOCK_DURATION}s = "
          f"{len(sequence)*BLOCK_DURATION:.0f}s")
    print(f"    {sum(1 for s in sequence if s>0)} comandos activos  "
          f"({sum(1 for s in sequence if s==1)} LEFT  "
          f"{sum(1 for s in sequence if s==2)} RIGHT  "
          f"{sum(1 for s in sequence if s==3)} FEET)")

    # Verificação de alinhamento
    print(f"\n  Verificação épocas [mi_start+{EPOCH_TMIN}s → mi_start+{EPOCH_TMAX}s]:")
    mi_rows = [(m[0], m[2]) for m in markers if m[1] == "mi_start"]
    all_ok  = True
    for t_mi, lbl in mi_rows:
        t_block_mi_start = round(t_mi, 6)
        t_block_end      = round(t_mi + MI_DURATION, 6)
        t_epoch_end      = round(t_mi + EPOCH_TMAX, 6)
        ok = t_epoch_end <= t_block_end
        if not ok:
            all_ok = False
            print(f"    ✗ label={lbl} epoch_end={t_epoch_end:.2f} > block_end={t_block_end:.2f}")
    if all_ok:
        print(f"    ✓ todas as {len(mi_rows)} épocas estão dentro dos blocos MI")

    # Guarda
    eeg_df = pd.DataFrame(eeg.T, columns=[f"ch_{i}" for i in range(8)])
    eeg_df["timestamp"] = times

    marker_df = pd.DataFrame(markers, columns=["timestamp", "event", "label"])

    out_dir = os.path.join(OUT_DIR, f"S{subject:03d}_demo")
    os.makedirs(out_dir, exist_ok=True)

    eeg_df.to_csv(os.path.join(out_dir, "eeg_raw.csv"), index=False)
    marker_df.to_csv(os.path.join(out_dir, "markers.csv"), index=False)
    with open(os.path.join(out_dir, "sfreq.txt"), "w") as f:
        f.write(str(sfreq))

    print(f"\n  → {out_dir}/")
    print(f"     eeg_raw.csv  {eeg_df.shape[0]} linhas × {eeg_df.shape[1]} colunas")
    print(f"     markers.csv  {marker_df.shape[0]} linhas")
    print(f"     sfreq.txt    {sfreq} Hz")


# ============================================================
# RUN
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)

for subject in SUBJECTS:
    build_session(subject)

print("\n\nDONE")
print(f"Sessões em: {OUT_DIR}/")
print(f"\nEstrutura de cada bloco ({BLOCK_DURATION}s):")
print(f"  [0s → {CUE_DURATION}s]        cue  — classificador ignora")
print(f"  [{CUE_DURATION}s → {BLOCK_DURATION}s]  MI real  — classificador usa [{CUE_DURATION+EPOCH_TMIN}s → {CUE_DURATION+EPOCH_TMAX}s]")

# ============================================================
# FILE: test_model.py
#
# Testa o modelo treinado em tempo real.
# A pessoa pensa numa classe, o modelo classifica,
# e depois pede validação (certo/errado) para calcular métricas.
#
# Uso: python test_model.py <session_path>
# Ex:  python test_model.py data/P001_20250521_143000
#
# PRÉ-PROCESSAMENTO IDÊNTICO AO TREINO (train_subject_model.py):
#   1. Referência média (average reference)
#   2. Filtro FIR bandpass 8-30 Hz  (fir_design="firwin", como MNE)
#   3. Filtro notch 25 Hz e 50 Hz
#   4. Janela de 4.0 s (= EPOCH_TMAX - EPOCH_TMIN = 4.5 - 0.5)
#   5. Pipeline CSP → Scaler → LDA (guardado no .pkl, não replicado aqui)
# ============================================================

import os
import sys
import time
import joblib
import numpy as np
from scipy.signal import firwin, sosfilt, sosfiltfilt, butter, iirnotch

import mne

from brainflow.board_shim import BoardShim, BrainFlowInputParams

from config import CONFIG


# ============================================================
# CONFIGURAÇÃO — deve coincidir com train_subject_model.py
# ============================================================

L_FREQ   = 8.0
H_FREQ   = 30.0
NOTCH_FREQS = [25.0, 50.0]

# Janela de classificação = EPOCH_TMAX - EPOCH_TMIN do treino
WINDOW_SEC = 4.0   # 4.5 - 0.5

# Padding extra para o filtro não ter artefactos de borda
# na região de interesse. Deve ser >= comprimento do filtro FIR / sfreq.
# 2 s é seguro para os filtros usados pelo MNE a 250 Hz.
FILTER_PAD_SEC = 2.0

# Tempo de preparação antes de cada trial (conta decrescente)
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
# PRÉ-PROCESSAMENTO — idêntico ao train_subject_model.py
# ============================================================

def make_fir_bandpass(l_freq, h_freq, sfreq):
    """
    Cria coeficientes FIR bandpass com firwin, igual ao MNE por omissão.
    Retorna coeficientes b para usar com sosfiltfilt via conversão.

    O MNE usa fir_design="firwin" com transition_bandwidth automático.
    Aqui replicamos com comprimento de filtro conservador (mesmo que o MNE
    use para sinais de 250 Hz com estes parâmetros de corte).
    """
    # Comprimento do filtro: MNE usa tipicamente ~0.34 s de dados para
    # 8 Hz com sfreq=250 → 85 amostras; arredondamos para garantir segurança.
    # Fórmula MNE: n_taps = int(round(0.34 * sfreq)) | sempre ímpar
    n_taps = int(round(0.34 * sfreq))
    if n_taps % 2 == 0:
        n_taps += 1

    b = firwin(n_taps, [l_freq, h_freq], pass_zero=False, fs=sfreq)
    return b


def make_notch_sos(freq, sfreq, quality=30.0):
    """
    Cria filtro notch IIR (biquad) como second-order sections.
    quality=30 é o valor por omissão do MNE para notch_filter.
    """
    b, a = iirnotch(freq, quality, fs=sfreq)
    # Converte para SOS para estabilidade numérica
    from scipy.signal import tf2sos
    sos = tf2sos(b, a)
    return sos


def preprocess_window(eeg_raw, sfreq, fir_b, notch_sos_list, n_need, n_total):
    """
    Aplica o mesmo pré-processamento do treino a uma janela de EEG em bruto.

    Parameters
    ----------
    eeg_raw        : np.ndarray (n_ch, n_samples_total)  — inclui padding
    sfreq          : float
    fir_b          : np.ndarray — coeficientes FIR bandpass
    notch_sos_list : list de np.ndarray — SOS de cada filtro notch
    n_need         : int — amostras da janela final (sem padding)
    n_total        : int — amostras totais capturadas (com padding)

    Returns
    -------
    window : np.ndarray (1, n_ch, n_need)  — pronto para clf.predict()
             ou None se não houver amostras suficientes
    """
    n_samples = eeg_raw.shape[1]

    if n_samples < n_need:
        return None

    # Usa as últimas n_total amostras (ou tudo se for menos)
    n_use = min(n_samples, n_total)
    eeg   = eeg_raw[:, -n_use:]

    # --- [1] Referência média (average reference) ---
    # Subtrai a média de todos os canais em cada instante de tempo,
    # exatamente como raw.set_eeg_reference("average") no MNE.
    eeg = eeg - eeg.mean(axis=0, keepdims=True)

    # --- [2] Filtro FIR bandpass 8-30 Hz (mesmo que MNE firwin) ---
    from scipy.signal import filtfilt
    eeg = filtfilt(fir_b, [1.0], eeg, axis=1)

    # --- [3] Filtro notch (25 Hz e 50 Hz) ---
    for sos in notch_sos_list:
        eeg = sosfiltfilt(sos, eeg, axis=1)

    # --- [4] Descarta padding — fica só a janela de classificação ---
    eeg_window = eeg[:, -n_need:]

    return eeg_window[np.newaxis, :, :]   # (1, n_ch, n_times)


# ============================================================
# LOAD MODELS
# ============================================================

def load_models(session_path):

    paths = {
        "gating":    os.path.join(session_path, "model_gating.pkl"),
        "axis":      os.path.join(session_path, "model_axis.pkl"),
        "direction": os.path.join(session_path, "model_direction.pkl"),
    }

    missing = [k for k, p in paths.items() if not os.path.exists(p)]

    if missing:
        raise FileNotFoundError(
            f"Modelos não encontrados em {session_path}: {missing}\n"
            f"Corre primeiro o main.py para treinar o modelo."
        )

    clf_gate = joblib.load(paths["gating"])
    clf_axis = joblib.load(paths["axis"])
    clf_dir  = joblib.load(paths["direction"])

    print(f"{GREEN}✓ Modelos carregados de {session_path}{RESET}")

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
# ACQUIRE + PRÉ-PROCESSAR + CLASSIFICAR
# ============================================================

def get_window(board, eeg_channels, sfreq,
               fir_b, notch_sos_list,
               window_sec, pad_sec):
    """
    Aguarda window_sec + pad_sec segundos, aplica pré-processamento
    idêntico ao treino e devolve (1, n_ch, n_times).

    O sinal bruto é em µV (BrainFlow devolve em µV).
    O treino converte para V (×1e-6) mas os CSP/Scaler/LDA
    aprenderam com V — por isso convertemos aqui também.
    """
    total_sec = window_sec + pad_sec
    n_need    = int(round(sfreq * window_sec))
    n_total   = int(round(sfreq * total_sec))

    # Limpa buffer antes de começar a contagem
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
    Corre um trial: conta decrescente → imagética → classificação → validação.
    Devolve (pred, correct).
    """
    # Conta decrescente
    for i in range(PREP_SEC, 0, -1):
        print(f"\r  {DIM}A começar em {i}...{RESET}  ", end="", flush=True)
        time.sleep(1.0)

    print(f"\r  {BOLD}{CYAN}PENSA: {SYMBOLS[target_label]} {NAMES[target_label]}{RESET}          ")
    print(f"  {DIM}(a classificar {WINDOW_SEC:.0f}s de sinal + {FILTER_PAD_SEC:.0f}s padding de filtro...){RESET}")

    # Adquire, pré-processa e classifica
    window = get_window(
        board, eeg_channels, sfreq,
        fir_b, notch_sos_list,
        WINDOW_SEC, FILTER_PAD_SEC
    )

    if window is None:
        print(f"  {RED}ERRO: sem dados EEG suficientes.{RESET}")
        return None, None

    pred, path = classify(window, clf_gate, clf_axis, clf_dir)

    # Mostra resultado
    print()
    print(f"  {'─' * 40}")
    for step in path:
        print(f"  {DIM}{step}{RESET}")
    print(f"  {'─' * 40}")
    print(f"  Classificação: {BOLD}{CYAN}{SYMBOLS[pred]} {NAMES[pred]}{RESET}")
    print(f"  {'─' * 40}")

    correct = (pred == target_label)

    if correct:
        print(f"  {GREEN}{BOLD}✓ CORRETO{RESET}")
    else:
        print(f"  {RED}{BOLD}✗ ERRADO{RESET}  (era {NAMES[target_label]})")

    print()
    val = input(
        f"  Concordas? [{GREEN}y{RESET}/{RED}n{RESET}/Enter=sim] "
    ).strip().lower()

    if val == "n":
        print(f"  Classes: {', '.join(f'{v}={k}' for k, v in NAMES.items())}")
        try:
            override = int(input("  Classe real (0/1/2/3): ").strip())
            correct  = (override == target_label)
            print(f"  {DIM}Corrigido para {NAMES.get(override, str(override))}{RESET}")
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
    print(f"{BOLD}RESULTADOS DA SESSÃO DE TESTE{RESET}")
    print("=" * 50)
    print(f"  Trials:   {n_total}")
    print(f"  Corretos: {n_correct}")
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
        print(f"  Enter    → próximo trial (classe aleatória)")
        print(f"  0/1/2/3  → escolhe a classe manualmente")
        print(f"  q        → terminar e ver métricas\n")

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

            print(f"\n  {BOLD}Prepara-te para pensar em: {SYMBOLS[target]} {NAMES[target]}{RESET}")

            pred, correct = run_trial(
                board, eeg_channels, sfreq,
                fir_b, notch_sos_list,
                clf_gate, clf_axis, clf_dir,
                target
            )

            if pred is not None:
                results.append((target, pred, correct))

    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}Interrompido.{RESET}")

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
            print("Nenhuma sessão com modelos encontrada em data/")
            print("Uso: python test_model.py data/P001_20250521_143000")
            sys.exit(1)

        session_path = sessions[-1]
        print(f"A usar sessão mais recente: {session_path}")
    else:
        session_path = sys.argv[1]

    main(session_path)
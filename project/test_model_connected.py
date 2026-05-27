# ============================================================
# FILE: test_model_connected.py
#
# Controlo do robô em tempo real via EEG + classificadores BCI.
#
# Estratégia de classificação:
#   - Janela deslizante de 2s com passo de 0.5s
#   - Voto maioritário sobre 3 classificações consecutivas
#   - Só envia comando ao robô se 2/3 classificações concordarem
#   - REST nunca envia comando (debounce natural)
#
# Latência típica: ~3.5s desde pensamento até comando
#   (2s janela + 2 × 0.5s passos + tempo de computação ~0ms)
#
# Uso: python test_model_connected.py <session_path> [serial_port]
# Ex:  python test_model_connected.py data/P001_20250521_143000 COM4
# ============================================================

import os
import sys
import time
import joblib
import collections
import numpy as np
import serial
from scipy.signal import butter, sosfiltfilt

from brainflow.board_shim import BoardShim, BrainFlowInputParams

from config import CONFIG


# ============================================================
# CONFIGURAÇÃO
# ============================================================

# Janela de classificação em segundos.
# 2s é o mínimo recomendado para CSP+LDA detectar ERD mu/beta.
# Deve ser <= EPOCH_TMAX - EPOCH_TMIN do treino (4s) — usamos 2s
# para reduzir latência mantendo fiabilidade.
WINDOW_SEC     = 2.0

# Passo do sliding window — quanto avança entre classificações.
# 0.5s → 3 classificações em ~3.5s de sinal total.
STEP_SEC       = 0.5

# Número de classificações no voto maioritário.
# Com VOTE_N=3 e VOTE_THRESHOLD=2: precisa de 2/3 para confirmar.
VOTE_N         = 1      # 3
VOTE_THRESHOLD = 1      # 2

# Padding para evitar artefactos de borda do filtro.
FILTER_PAD_SEC = 1.0

# Filtro — igual ao treino
L_FREQ = 8.0
H_FREQ = 30.0

# Serial do robô
ROBOT_BAUD = 9600

# Mapeamento classe BCI → comando Arduino
# Formato: string enviada via Serial + '\n'
# Arduino aceita: +1 -1 +2 -2 +3 -3 +4 -4
# Altera aqui conforme a mecânica do robô
COMMAND_MAP = {
    0: None,     # REST  → sem comando
    1: "LEFT",   # LEFT  → servo activo recua
    2: "RIGHT",  # RIGHT → servo activo avança
    3: "FEET",   # FEET  → cicla servo activo (0→1→2→3→0)
}

SYMBOLS = {0: "+", 1: "←", 2: "→", 3: "↓"}
NAMES   = {0: "REST", 1: "LEFT", 2: "RIGHT", 3: "FEET"}

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ============================================================
# FILTRO (igual ao test_model.py)
# ============================================================

def make_bandpass(l_freq, h_freq, sfreq, order=8):
    nyq = sfreq / 2.0
    sos = butter(order, [l_freq / nyq, h_freq / nyq], btype="band", output="sos")
    return sos

def apply_bandpass(eeg, sos):
    return sosfiltfilt(sos, eeg, axis=1)


# ============================================================
# MODELOS
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
            f"Modelos não encontrados em '{session_path}': {missing}"
        )
    clf_gate = joblib.load(paths["gating"])
    clf_axis = joblib.load(paths["axis"])
    clf_dir  = joblib.load(paths["direction"])
    print(f"{GREEN}✓ Modelos carregados{RESET}")
    return clf_gate, clf_axis, clf_dir


# ============================================================
# BOARD
# ============================================================

def start_board():
    params = BrainFlowInputParams()
    params.serial_port = CONFIG["serial_port"]
    board    = BoardShim(CONFIG["board_id"], params)
    board_id = CONFIG["board_id"]
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sfreq        = BoardShim.get_sampling_rate(board_id)
    board.prepare_session()
    board.start_stream()
    time.sleep(2)
    print(f"{GREEN}✓ EEG ligado | {len(eeg_channels)} canais | {sfreq}Hz{RESET}")
    return board, eeg_channels, sfreq

def stop_board(board):
    try:
        board.stop_stream()
        board.release_session()
    except Exception:
        pass


# ============================================================
# ROBOT SERIAL
# ============================================================

def open_robot(port, baud=ROBOT_BAUD):
    try:
        rob = serial.Serial(port, baud, timeout=1)
        time.sleep(2)   # Arduino reset após abertura da porta
        # Lê a mensagem de arranque do Arduino
        if rob.in_waiting:
            print(f"  Arduino: {rob.readline().decode().strip()}")
        print(f"{GREEN}✓ Robô ligado em {port}{RESET}")
        return rob
    except serial.SerialException as e:
        raise RuntimeError(f"Não foi possível abrir {port}: {e}")

def send_command(rob, cmd_str):
    """Envia um comando ao Arduino e lê a resposta."""
    if rob is None or cmd_str is None:
        return
    try:
        rob.write((cmd_str + "\n").encode())
        time.sleep(0.05)
        if rob.in_waiting:
            resp = rob.readline().decode().strip()
            print(f"  {DIM}Arduino: {resp}{RESET}")
    except serial.SerialException as e:
        print(f"  {RED}Erro serial: {e}{RESET}")

def close_robot(rob):
    if rob and rob.is_open:
        rob.close()


# ============================================================
# CLASSIFICAÇÃO — JANELA ÚNICA
# ============================================================

def classify_window(window, clf_gate, clf_axis, clf_dir):
    """
    Cascata de 3 classificadores numa janela (1, n_ch, n_times).
    Devolve label final (0-3).
    """
    pred_gate = clf_gate.predict(window)[0]
    if pred_gate == 0:
        return 0                            # REST

    pred_axis = clf_axis.predict(window)[0]
    if pred_axis == 1:
        return 3                            # FEET

    pred_dir = clf_dir.predict(window)[0]
    return int(pred_dir)                    # LEFT(1) ou RIGHT(2)


# ============================================================
# SLIDING WINDOW + VOTO MAIORITÁRIO
# ============================================================

def sliding_vote(board, eeg_channels, sfreq, sos,
                 clf_gate, clf_axis, clf_dir,
                 window_sec=WINDOW_SEC,
                 step_sec=STEP_SEC,
                 vote_n=VOTE_N,
                 pad_sec=FILTER_PAD_SEC):
    """
    Recolhe `vote_n` classificações com janelas deslizantes.

    Timeline para vote_n=3, window=2s, step=0.5s:
      t=0.0  flush buffer
      t=2.0  1ª janela pronta → classifica
      t=2.5  2ª janela pronta → classifica
      t=3.0  3ª janela pronta → classifica → voto → comando
      Latência total: ~3.0s + pad

    O buffer do BrainFlow é acumulativo — não fazemos flush entre
    janelas, apenas calculamos o offset de amostras correcto.

    Returns
    -------
    winner : int — label com maioria (ou -1 se empate sem maioria)
    votes  : list[int] — as 3 classificações individuais
    """
    n_win  = int(sfreq * window_sec)
    n_step = int(sfreq * step_sec)
    n_pad  = int(sfreq * pad_sec)

    # Total de amostras necessárias:
    # padding + janela base + (vote_n - 1) passos
    n_need = n_pad + n_win + (vote_n - 1) * n_step

    # Flush e aguarda amostras suficientes
    board.get_board_data()
    time.sleep(n_need / sfreq)

    data = board.get_board_data()

    if data.shape[1] < n_need:
        return -1, []

    eeg = data[eeg_channels, :]          # (n_ch, n_samples)

    # Garante que temos pelo menos n_need amostras (pega as mais recentes)
    if eeg.shape[1] > n_need:
        eeg = eeg[:, -n_need:]

    # Filtra tudo de uma vez (mais eficiente e evita artefactos de borda)
    eeg_filt = apply_bandpass(eeg, sos)

    # Descarta o padding inicial
    eeg_filt = eeg_filt[:, n_pad:]       # (n_ch, n_win + (vote_n-1)*n_step)

    votes = []
    for i in range(vote_n):
        start = i * n_step
        end   = start + n_win
        if end > eeg_filt.shape[1]:
            break
        window = eeg_filt[:, start:end][np.newaxis, :, :]   # (1, n_ch, n_win)
        pred   = classify_window(window, clf_gate, clf_axis, clf_dir)
        votes.append(pred)

    if not votes:
        return -1, []

    # Voto maioritário
    counter = collections.Counter(votes)
    winner, count = counter.most_common(1)[0]

    if count >= VOTE_THRESHOLD:
        return winner, votes
    else:
        return -1, votes    # Sem consenso → não actua


# ============================================================
# LOOP PRINCIPAL
# ============================================================

def run(session_path, robot_port):

    print("\n" + "=" * 55)
    print(f"{BOLD}INSERT-BRAIN — CONTROLO CONTÍNUO{RESET}")
    print(f"  Sessão:  {session_path}")
    print(f"  Robô:    {robot_port}")
    print(f"  Janela:  {WINDOW_SEC}s  |  Passo: {STEP_SEC}s  |  Votos: {VOTE_N}")
    print("=" * 55 + "\n")

    clf_gate, clf_axis, clf_dir = load_models(session_path)
    board, eeg_channels, sfreq  = start_board()
    rob                         = open_robot(robot_port)
    sos                         = make_bandpass(L_FREQ, H_FREQ, sfreq)

    # Estatísticas da sessão
    stats = collections.defaultdict(int)
    n_total = 0
    n_no_consensus = 0

    print(f"\n{BOLD}A classificar continuamente. Ctrl+C para parar.{RESET}")
    print(f"{DIM}Mapeamento: LEFT→servo activo recua  RIGHT→servo activo avança  FEET→cicla servo  REST→nenhum{RESET}\n")

    try:
        while True:

            winner, votes = sliding_vote(
                board, eeg_channels, sfreq, sos,
                clf_gate, clf_axis, clf_dir
            )

            n_total += 1
            ts = time.strftime("%H:%M:%S")

            if winner == -1:
                n_no_consensus += 1
                votes_str = " ".join(NAMES.get(v, "?") for v in votes)
                print(f"[{ts}]  {YELLOW}sem consenso{RESET}  [{votes_str}]")
                continue

            stats[winner] += 1
            cmd = COMMAND_MAP.get(winner)
            votes_str = " ".join(NAMES.get(v, "?") for v in votes)

            if winner == 0:
                # REST — não actua
                print(f"[{ts}]  {DIM}REST{RESET}  [{votes_str}]")
            else:
                print(
                    f"[{ts}]  {BOLD}{CYAN}{SYMBOLS[winner]} {NAMES[winner]}{RESET}"
                    f"  [{votes_str}]  → {GREEN}{cmd}{RESET}"
                )
                send_command(rob, cmd)

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Parado pelo utilizador.{RESET}")

    finally:
        stop_board(board)
        close_robot(rob)

        # Sumário
        print("\n" + "=" * 55)
        print(f"{BOLD}SUMÁRIO DA SESSÃO{RESET}")
        print(f"  Ciclos totais:    {n_total}")
        print(f"  Sem consenso:     {n_no_consensus}")
        print(f"  Comandos enviados:")
        for label, name in NAMES.items():
            if label == 0:
                continue
            n = stats[label]
            cmd = COMMAND_MAP.get(label, "?")
            print(f"    {name:<8} ({cmd})  →  {n}×")
        print("=" * 55 + "\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Session path
    if len(sys.argv) >= 2:
        session_path = sys.argv[1]
    else:
        sessions = sorted([
            os.path.join("data", d)
            for d in os.listdir("data")
            if os.path.isdir(os.path.join("data", d))
            and os.path.exists(os.path.join("data", d, "model_gating.pkl"))
        ])
        if not sessions:
            print("Nenhuma sessão com modelos encontrada em data/")
            print("Uso: python test_model_connected.py <session_path> [serial_port]")
            sys.exit(1)
        session_path = sessions[-1]
        print(f"A usar sessão mais recente: {session_path}")

    # Robot serial port
    if len(sys.argv) >= 3:
        robot_port = sys.argv[2]
    else:
        robot_port = input("Porta série do robô (ex: COM4 ou /dev/ttyUSB1): ").strip()

    run(session_path, robot_port)

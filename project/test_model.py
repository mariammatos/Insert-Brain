# ============================================================
# FILE: test_model.py
#
# Testa o modelo treinado em tempo real.
# A pessoa pensa numa classe, o modelo classifica,
# e depois pede validação (certo/errado) para calcular métricas.
#
# Uso: python test_model.py <session_path>
# Ex:  python test_model.py data/P001_20250521_143000
# ============================================================

import os
import sys
import json
import time
import joblib
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams

from config import CONFIG


# ============================================================
# CONFIGURAÇÃO
# ============================================================

# Janela de classificação em segundos
WINDOW_SEC  = 2.0

# Tempo de preparação antes de cada trial (conta decrescente)
PREP_SEC    = 3

# Símbolos visuais no terminal
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

# Cores ANSI para terminal
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ============================================================
# LOAD MODELS
# ============================================================

def load_models(session_path):

    paths = {
        "gating":    os.path.join(session_path, "model_gating.pkl"),
        "axis":      os.path.join(session_path, "model_direction.pkl"),
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

    print(f"{GREEN}✓ Board ligado | {len(eeg_channels)} canais | {sfreq}Hz{RESET}\n")

    return board, eeg_channels, timestamp_channel, sfreq


def stop_board(board):
    board.stop_stream()
    board.release_session()


# ============================================================
# CLASSIFY WINDOW
# ============================================================

def get_window(board, eeg_channels, sfreq, window_sec):
    """
    Aguarda window_sec segundos e devolve a janela de EEG
    no formato (1, n_ch, n_times) para o classificador.
    """

    # Limpa buffer antes de começar
    board.get_board_data()

    time.sleep(window_sec)

    data = board.get_board_data()

    if data.shape[1] == 0:
        return None

    eeg    = data[eeg_channels, :]   # (n_ch, n_samples)
    n_need = int(sfreq * window_sec)

    # Usa as últimas n_need amostras
    if eeg.shape[1] >= n_need:
        eeg = eeg[:, -n_need:]
    
    return eeg[np.newaxis, :, :]     # (1, n_ch, n_times)


def classify(window, clf_gate, clf_axis, clf_dir):
    """
    Cascata de classificadores.
    Devolve (pred_final, caminho) onde caminho descreve as decisões.
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

def run_trial(board, eeg_channels, sfreq, clf_gate, clf_axis, clf_dir, target_label):
    """
    Corre um trial: conta decrescente → imagética → classificação → validação.
    Devolve (pred, correct).
    """

    # Conta decrescente
    for i in range(PREP_SEC, 0, -1):
        print(f"\r  {DIM}A começar em {i}...{RESET}  ", end="", flush=True)
        time.sleep(1.0)

    print(f"\r  {BOLD}{CYAN}PENSA: {SYMBOLS[target_label]} {NAMES[target_label]}{RESET}          ")
    print(f"  {DIM}(a classificar {WINDOW_SEC:.0f}s de sinal...){RESET}")

    # Adquire e classifica
    window = get_window(board, eeg_channels, sfreq, WINDOW_SEC)

    if window is None:
        print(f"  {RED}ERRO: sem dados EEG.{RESET}")
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

    # Validação manual (opcional — pode corrigir se o modelo errou algo óbvio)
    print()
    val = input(
        f"  Concordas? [{GREEN}y{RESET}/{RED}n{RESET}/Enter=sim] "
    ).strip().lower()

    if val == "n":
        # A pessoa discorda — pede a classe real
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
    """
    Calcula e imprime métricas no final da sessão de teste.
    results: list of (target, pred, correct)
    """

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

    # Por classe
    print(f"\n  {'Classe':<10} {'Trials':>6} {'Corretos':>8} {'Acc':>6}")
    print(f"  {'─'*34}")

    for label, name in NAMES.items():
        cls_trials   = [r for r in results if r[0] == label]
        cls_correct  = sum(r[2] for r in cls_trials)
        cls_n        = len(cls_trials)

        if cls_n == 0:
            continue

        cls_acc = cls_correct / cls_n
        color   = GREEN if cls_acc >= 0.7 else YELLOW if cls_acc >= 0.5 else RED

        print(
            f"  {name:<10} {cls_n:>6} {cls_correct:>8} "
            f"{color}{cls_acc*100:>5.1f}%{RESET}"
        )

    # Kappa simples
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

    # Carrega modelos
    clf_gate, clf_axis, clf_dir = load_models(session_path)

    # Liga board
    board, eeg_channels, timestamp_channel, sfreq = start_board()

    results = []

    classes = list(NAMES.keys())   # [0, 1, 2, 3]

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
                # Aleatório — exclui REST se já houve suficientes
                target = int(np.random.choice(classes))

            print(f"\n  {BOLD}Prepara-te para pensar em: {SYMBOLS[target]} {NAMES[target]}{RESET}")

            pred, correct = run_trial(
                board, eeg_channels, sfreq,
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
        # Usa sessão mais recente automaticamente
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
# ============================================================
# FILE: retrain.py
# ============================================================
#
# Retreina os modelos a partir de dados já guardados numa sessão.
#
# Uso:
#   python retrain.py
#   python retrain.py data/P001_20250521_143000
#
# ============================================================

import os
import sys

from training.train_subject_model import train_subject_model


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Aceita o caminho como argumento ou pede interactivamente
    if len(sys.argv) >= 2:
        session_path = sys.argv[1]
    else:
        session_path = input("Session path: ").strip()

    # Validação básica
    required_files = ["eeg_raw.csv", "markers.csv", "metadata.json"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(session_path, f))]

    if missing:
        print(f"\nERRO: Ficheiros em falta em '{session_path}':")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)

    # Treino
    print(f"\nA retreinar modelos para a sessão: {session_path}")
    clf_gate, clf_axis, clf_dir, report = train_subject_model(session_path)

    print(f"\nModelos guardados em: {session_path}")
    print("\nRetraining completo.")
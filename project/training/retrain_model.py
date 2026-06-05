# ============================================================
# FILE: training/retrain_model.py
#
# Retrains models from data already saved in a session folder.
#
# Usage:
#   python retrain_model.py
#   python retrain_model.py data/P001_20250521_143000
# ============================================================

import os
import sys

from training.train_subject_model import train_subject_model


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Accept path as argument or prompt interactively
    if len(sys.argv) >= 2:
        session_path = sys.argv[1]
    else:
        session_path = input("Session path: ").strip()

    # Basic validation
    required_files = ["eeg_raw.csv", "markers.csv", "metadata.json"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(session_path, f))]

    if missing:
        print(f"\nERROR: Missing files in '{session_path}':")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)

    # Train
    print(f"\nRetraining models for session: {session_path}")
    clf_gate, clf_axis, clf_dir, report = train_subject_model(session_path)

    print(f"\nModels saved to: {session_path}")
    print("\nRetraining complete.")
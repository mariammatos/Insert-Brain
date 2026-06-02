"""
train_multi_session_model.py

Usage:
    python train_multi_session_model.py <output_dir> <session1> <session2> [session3 ...]
    python project/training/train_multi_session_model.py data/combined_Afonso2 --name Afonso2
"""

import numpy as np
import joblib
import os

from train_subject_model import (
    load_session_data,
    build_mne_raw,
    preprocess_raw,
    build_epochs,
    train_pipeline,
)

def train_multi_session_model(session_paths, output_dir):

    X_gate_all = []
    y_gate_all = []

    X_axis_all = []
    y_axis_all = []

    X_dir_all = []
    y_dir_all = []

    for session_path in session_paths:

        print(f"\nLoading {session_path}")

        eeg_df, markers, metadata = load_session_data(session_path)

        raw, eeg_start_unix = build_mne_raw(eeg_df, metadata)
        raw = preprocess_raw(raw)

        sfreq = metadata["sampling_rate"]

        classes = metadata["classes"]

        rest_label  = classes["REST"]["label"]
        left_label  = classes["LEFT"]["label"]
        right_label = classes["RIGHT"]["label"]
        feet_label  = classes["FEET"]["label"]

        # -------------------------
        # GATING
        # -------------------------

        gating_filter = {
            rest_label: 0,
            left_label: 1,
            right_label: 1,
            feet_label: 1,
        }

        X_gate, y_gate = build_epochs(
            raw,
            markers,
            sfreq,
            gating_filter,
            eeg_start_unix
        )

        X_gate_all.append(X_gate)
        y_gate_all.append(y_gate)

        # -------------------------
        # AXIS
        # -------------------------

        axis_filter = {
            left_label: 0,
            right_label: 0,
            feet_label: 1,
        }

        X_axis, y_axis = build_epochs(
            raw,
            markers,
            sfreq,
            axis_filter,
            eeg_start_unix
        )

        X_axis_all.append(X_axis)
        y_axis_all.append(y_axis)

        # -------------------------
        # DIRECTION
        # -------------------------

        dir_filter = {
            left_label: left_label,
            right_label: right_label,
        }

        X_dir, y_dir = build_epochs(
            raw,
            markers,
            sfreq,
            dir_filter,
            eeg_start_unix
        )

        X_dir_all.append(X_dir)
        y_dir_all.append(y_dir)

    # ==================================
    # MERGE ALL DATA
    # ==================================

    X_gate = np.concatenate(X_gate_all, axis=0)
    y_gate = np.concatenate(y_gate_all)

    X_axis = np.concatenate(X_axis_all, axis=0)
    y_axis = np.concatenate(y_axis_all)

    X_dir = np.concatenate(X_dir_all, axis=0)
    y_dir = np.concatenate(y_dir_all)

    print("\nCombined dataset:")
    print("Gating:", X_gate.shape)
    print("Axis:", X_axis.shape)
    print("Direction:", X_dir.shape)

    clf_gate, _ = train_pipeline(X_gate, y_gate, "GATING")
    clf_axis, _ = train_pipeline(X_axis, y_axis, "AXIS")
    clf_dir, _ = train_pipeline(X_dir, y_dir, "DIRECTION")

    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(clf_gate, os.path.join(output_dir, "model_gating.pkl"))
    joblib.dump(clf_axis, os.path.join(output_dir, "model_axis.pkl"))
    joblib.dump(clf_dir, os.path.join(output_dir, "model_direction.pkl"))

    print("\nModels saved.")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import sys
    import os
    import glob
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "output_dir",
        help="Directory where models will be saved"
    )

    parser.add_argument(
        "sessions",
        nargs="*",
        help="Explicit session folders"
    )

    parser.add_argument(
        "--name",
        help="Subject name/prefix used to automatically find sessions"
    )

    args = parser.parse_args()

    sessions = []

    if args.name:

        prefix = os.path.basename(args.name)

        search_pattern = os.path.join(
            os.path.dirname(args.name) or "data",
            f"{prefix}_*"
        )

        sessions = [
            p for p in glob.glob(search_pattern)
            if os.path.isdir(p)
        ]

        sessions.sort()

        print(f"\nFound {len(sessions)} sessions:")
        for s in sessions:
            print(f"  {s}")

    else:
        sessions = args.sessions

    if len(sessions) < 2:
        print(
            "Need at least 2 session folders.\n"
            "Examples:\n"
            "  python train_multi_session_model.py output session1 session2\n"
            "  python train_multi_session_model.py output --name Afonso2"
        )
        sys.exit(1)

    train_multi_session_model(
        session_paths=sessions,
        output_dir=args.output_dir
    )
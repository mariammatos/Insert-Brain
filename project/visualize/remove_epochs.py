# ============================================================
# FILE: visualize/remove_epochs.py
#
# Removes specific trials from a session's raw CSV files and
# saves a cleaned copy to a new folder (originals untouched).
#
# - Removes EEG samples for the trial window (mi_start → mi_end)
#   and stitches the remaining signal together (timestamps adjusted)
# - Removes the corresponding cue_on / mi_start / mi_end marker rows
# - Saves cleaned eeg_raw.csv, markers.csv, metadata.json to
#   <session>_cleaned/ (or a custom output path)
#
# Usage:
#   python remove_epochs.py <session_path> <class_label> <epoch_index> [epoch_index ...]
#
# Examples:
#   # Remove epoch 7 (1-based) of LEFT (label=1):
#   python remove_epochs.py data/P001_20250521_143000 LEFT 7
#
#   # Remove epochs 2 and 5 of RIGHT, and epoch 3 of FEET:
#   python remove_epochs.py data/P001_20250521_143000 RIGHT 2 5 -- FEET 3
#
# Epoch indices are 1-based (as shown in browse_epochs.py).
# ============================================================

import os
import sys
import json
import shutil
import argparse
import numpy as np
import pandas as pd


# ============================================================
# PARAMETERS — must match explore_session.py / browse_epochs.py
# ============================================================

EPOCH_TMIN = 0.5   # seconds before mi_start used as epoch onset
EPOCH_TMAX = 4.5   # seconds after mi_start used as epoch end


# ============================================================
# HELPERS
# ============================================================

def load_session(session_path):
    eeg_df   = pd.read_csv(os.path.join(session_path, "eeg_raw.csv"))
    markers  = pd.read_csv(os.path.join(session_path, "markers.csv"))
    with open(os.path.join(session_path, "metadata.json")) as f:
        metadata = json.load(f)
    return eeg_df, markers, metadata


def label_for_class(class_name, metadata):
    """Return the integer label for a class name."""
    classes = metadata.get("classes", {})
    name_upper = class_name.upper()
    for k, v in classes.items():
        if k.upper() == name_upper:
            return v["label"]
    raise ValueError(
        f"Class '{class_name}' not found in metadata. "
        f"Available: {list(classes.keys())}"
    )


def get_mi_start_times(markers, label):
    """
    Return sorted list of mi_start timestamps for a given label.
    Falls back to unique label rows if mi_start not present.
    """
    if "mi_start" in markers["event"].values:
        rows = markers[
            (markers["event"] == "mi_start") &
            (markers["label"] == label)
        ].sort_values("timestamp")
    else:
        rows = markers[
            markers["label"] == label
        ].drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    return rows["timestamp"].tolist()


def get_trial_window(mi_start_ts, markers):
    """
    Return (t_start, t_end) for the trial.
    t_start = timestamp of cue_on before this mi_start (or mi_start - EPOCH_TMIN)
    t_end   = timestamp of mi_end after this mi_start (or mi_start + EPOCH_TMAX)
    """
    mi_end_rows = markers[
        (markers["event"] == "mi_end") &
        (markers["timestamp"] > mi_start_ts)
    ].sort_values("timestamp")

    t_end = mi_end_rows.iloc[0]["timestamp"] if len(mi_end_rows) > 0 else mi_start_ts + EPOCH_TMAX

    cue_rows = markers[
        (markers["event"] == "cue_on") &
        (markers["timestamp"] < mi_start_ts)
    ].sort_values("timestamp")

    t_start = cue_rows.iloc[-1]["timestamp"] if len(cue_rows) > 0 else mi_start_ts - EPOCH_TMIN

    return t_start, t_end


# ============================================================
# CORE REMOVAL
# ============================================================

def remove_trials(session_path, removals, output_path=None):
    """
    removals: list of (class_name, epoch_1based_index) tuples
              e.g. [("LEFT", 7), ("RIGHT", 2)]
    """

    print(f"\nLoading session: {session_path}")
    eeg_df, markers, metadata = load_session(session_path)

    windows_to_remove = []   # list of (t_start, t_end, class_name, epoch_idx, mi_ts)

    for class_name, epoch_idx in removals:
        label = label_for_class(class_name, metadata)
        mi_starts = get_mi_start_times(markers, label)

        if len(mi_starts) == 0:
            print(f"  ⚠ No epochs found for class '{class_name}' (label={label}). Skipping.")
            continue

        idx_0based = epoch_idx - 1
        if idx_0based < 0 or idx_0based >= len(mi_starts):
            raise IndexError(
                f"Epoch index {epoch_idx} out of range for class '{class_name}' "
                f"(has {len(mi_starts)} epochs)."
            )

        mi_ts = mi_starts[idx_0based]
        t_start, t_end = get_trial_window(mi_ts, markers)
        windows_to_remove.append((t_start, t_end, class_name, epoch_idx, mi_ts))
        print(f"  Marked for removal: {class_name} epoch {epoch_idx}  "
              f"(mi_start={mi_ts:.3f}, window={t_start:.3f} → {t_end:.3f})")

    if not windows_to_remove:
        print("Nothing to remove.")
        return

    # ---- Clean EEG ----
    print("\n  Cleaning EEG samples...")
    ts = eeg_df["timestamp"].values
    keep_mask = np.ones(len(eeg_df), dtype=bool)

    for t_start, t_end, *_ in windows_to_remove:
        keep_mask &= ~((ts >= t_start) & (ts <= t_end))

    removed_samples = (~keep_mask).sum()
    print(f"  Removed {removed_samples} EEG samples "
          f"({removed_samples / metadata['sampling_rate']:.2f} s)")

    eeg_clean = eeg_df[keep_mask].copy().reset_index(drop=True)

    # Stitch timestamps: shift each segment after a removed window so time is continuous
    ts_clean    = eeg_clean["timestamp"].values.copy()
    dt_nominal  = 1.0 / metadata["sampling_rate"]

    for i in range(1, len(ts_clean)):
        gap = ts_clean[i] - ts_clean[i - 1]
        if gap > dt_nominal * 2.5:
            shift = gap - dt_nominal
            ts_clean[i:] -= shift
            print(f"  Stitched gap of {gap:.4f}s at sample {i}")

    eeg_clean["timestamp"] = ts_clean

    # ---- Clean markers ----
    print("\n  Cleaning markers...")
    markers_clean = markers.copy()

    for t_start, t_end, *_ in windows_to_remove:
        markers_clean = markers_clean[
            ~((markers_clean["timestamp"] >= t_start) &
              (markers_clean["timestamp"] <= t_end))
        ]

    removed_markers = len(markers) - len(markers_clean)
    print(f"  Removed {removed_markers} marker rows")

    # Shift marker timestamps to match stitched EEG.
    # Process in reverse chronological order to avoid compounding offsets.
    sorted_windows = sorted(windows_to_remove, key=lambda x: x[0], reverse=True)
    mk_ts = markers_clean["timestamp"].values.copy()

    for t_start, t_end, *_ in sorted_windows:
        window_dur = t_end - t_start + dt_nominal
        after_mask = mk_ts > t_end
        mk_ts[after_mask] -= window_dur

    markers_clean = markers_clean.copy()
    markers_clean["timestamp"] = mk_ts

    # ---- Save ----
    if output_path is None:
        output_path = session_path.rstrip("/\\") + "_cleaned"

    os.makedirs(output_path, exist_ok=True)

    shutil.copy(
        os.path.join(session_path, "metadata.json"),
        os.path.join(output_path, "metadata.json")
    )

    eeg_clean.to_csv(os.path.join(output_path, "eeg_raw.csv"), index=False)
    markers_clean.to_csv(os.path.join(output_path, "markers.csv"), index=False)

    print(f"\n  ✓ Cleaned session saved to: {output_path}")
    print(f"    EEG rows : {len(eeg_df)} → {len(eeg_clean)}")
    print(f"    Markers  : {len(markers)} → {len(markers_clean)}")

    print("\n  Removed trials:")
    for _, _, class_name, epoch_idx, mi_ts in windows_to_remove:
        print(f"    {class_name} epoch {epoch_idx}  (mi_start={mi_ts:.3f})")


# ============================================================
# CLI
# ============================================================

def parse_args():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    session_path = sys.argv[1]
    rest         = sys.argv[2:]

    removals = []
    segments = " ".join(rest).split("--")

    for seg in segments:
        parts = seg.strip().split()
        if not parts:
            continue
        class_name = parts[0]
        for idx_str in parts[1:]:
            removals.append((class_name, int(idx_str)))

    return session_path, removals


if __name__ == "__main__":
    session_path, removals = parse_args()
    remove_trials(session_path, removals)
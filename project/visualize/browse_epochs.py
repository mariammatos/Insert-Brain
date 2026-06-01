# ============================================================
# FILE: browse_epochs.py
#
# Interactive viewer: browse filtered epochs one by one.
# Shows all channels for a single epoch at a time.
# Navigate with buttons or keyboard arrows.
#
# Usage:
#   python browse_epochs.py
#   python browse_epochs.py data/my_session_folder
# ============================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
import mne

mne.set_log_level("WARNING")

# ============================================================
# PARAMETERS — match your session setup
# ============================================================

L_FREQ      = 8.0
H_FREQ      = 30.0
EPOCH_TMIN  = 0.5
EPOCH_TMAX  = 4.5

CHANNEL_NAMES = ["FCz", "P3", "CP4", "CP3", "P4", "C3", "FC4", "FC3"]

CLASS_COLORS = {
    "REST":  "#888888",
    "LEFT":  "#e74c3c",
    "RIGHT": "#3498db",
    "FEET":  "#2ecc71",
}

# ============================================================
# LOAD / BUILD / FILTER  (same as explore_session.py)
# ============================================================

def load_session(session_path):
    eeg_df   = pd.read_csv(os.path.join(session_path, "eeg_raw.csv"))
    markers  = pd.read_csv(os.path.join(session_path, "markers.csv"))
    with open(os.path.join(session_path, "metadata.json")) as f:
        metadata = json.load(f)
    return eeg_df, markers, metadata


def build_raw(eeg_df, metadata):
    sfreq    = metadata["sampling_rate"]
    n_ch     = len(metadata["eeg_channels"])
    ch_names = CHANNEL_NAMES[:n_ch]

    eeg_cols = [c for c in eeg_df.columns if c.startswith("ch_")]
    data     = eeg_df[eeg_cols].values.T * 1e-6   # µV → V

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_ch)
    raw  = mne.io.RawArray(data, info, verbose=False)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore", verbose=False)

    eeg_start_unix = float(eeg_df["timestamp"].iloc[0])
    raw.set_meas_date(eeg_start_unix)
    return raw, eeg_start_unix, sfreq


def filter_raw(raw):
    raw_f = raw.copy().filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)
    raw_f.notch_filter(freqs=[25, 50], verbose=False)
    return raw_f


def build_epochs_per_class(raw_f, markers, sfreq, eeg_start_unix, metadata):
    classes = metadata.get("classes", {})
    label_to_name = {v.get("label"): k for k, v in classes.items()}

    if "mi_start" in markers["event"].values:
        cue_markers = markers[markers["event"] == "mi_start"].copy()
    else:
        cue_markers = markers[markers["label"] >= 0].copy()
        cue_markers = cue_markers.drop_duplicates(subset=["label"])

    cue_markers = cue_markers[cue_markers["label"].isin(label_to_name.keys())]
    cue_markers = cue_markers[cue_markers["label"] >= 0]

    if len(cue_markers) == 0:
        raise ValueError("No motor imagery markers found.")

    events = []
    for _, row in cue_markers.iterrows():
        t_rel  = row["timestamp"] - eeg_start_unix
        sample = int(round(t_rel * sfreq))
        if t_rel < 0 or sample >= raw_f.n_times:
            continue
        events.append([sample, 0, int(row["label"])])

    events    = np.array(events, dtype=int)
    event_map = {name: info["label"] for name, info in classes.items() if info["label"] >= 0}

    epochs = mne.Epochs(
        raw_f, events,
        event_id=event_map,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=None, preload=True, verbose=False
    )

    epochs_per_class = {}
    for name, label in event_map.items():
        try:
            ep = epochs[name]
            if len(ep) > 0:
                epochs_per_class[name] = ep
                print(f"  {name:>8}: {len(ep)} epochs")
        except KeyError:
            pass

    return epochs_per_class


# ============================================================
# INTERACTIVE VIEWER
# ============================================================

class EpochBrowser:
    """
    Browse epochs one by one across all classes.
    Each epoch shows all channels stacked in a grid.
    """

    def __init__(self, epochs_per_class):
        # Build a flat list of (class_name, epoch_index) entries
        self.epochs_per_class = epochs_per_class
        self.class_names      = list(epochs_per_class.keys())

        # Current state
        self.current_class = self.class_names[0]
        self.current_idx   = 0

        self._build_figure()
        self._draw()
        plt.show()

    # ----------------------------------------------------------

    def _n_epochs(self, cls=None):
        cls = cls or self.current_class
        return len(self.epochs_per_class[cls])

    def _build_figure(self):
        n_ch = len(self.epochs_per_class[self.current_class].ch_names)
        n_rows = (n_ch + 3) // 4   # 4 channels per row
        n_cols = min(n_ch, 4)

        self.fig = plt.figure(figsize=(16, 3 * n_rows + 2))
        self.fig.patch.set_facecolor("#1a1a2e")

        # Main title area
        self.title_ax = self.fig.add_axes([0.0, 0.93, 1.0, 0.07])
        self.title_ax.axis("off")
        self.title_ax.set_facecolor("#1a1a2e")
        self.title_text = self.title_ax.text(
            0.5, 0.5, "",
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            color="white",
            transform=self.title_ax.transAxes
        )

        # Channel axes grid
        gs = gridspec.GridSpec(
            n_rows, n_cols,
            left=0.06, right=0.98,
            top=0.90, bottom=0.14,
            hspace=0.55, wspace=0.35
        )
        self.ch_axes = []
        for i in range(n_ch):
            ax = self.fig.add_subplot(gs[i // n_cols, i % n_cols])
            ax.set_facecolor("#0f0f23")
            for spine in ax.spines.values():
                spine.set_color("#444466")
            ax.tick_params(colors="#aaaacc", labelsize=7)
            self.ch_axes.append(ax)

        # Hide any leftover axes slots
        total_slots = n_rows * n_cols
        for j in range(n_ch, total_slots):
            ax = self.fig.add_subplot(gs[j // n_cols, j % n_cols])
            ax.set_visible(False)

        # ---- Navigation buttons ----
        btn_h, btn_w = 0.045, 0.10
        btn_y = 0.025

        ax_prev  = self.fig.add_axes([0.38, btn_y, btn_w, btn_h])
        ax_next  = self.fig.add_axes([0.52, btn_y, btn_w, btn_h])

        # Class selector buttons (one per class)
        n_cls  = len(self.class_names)
        cls_w  = 0.08
        gap    = 0.01
        total  = n_cls * cls_w + (n_cls - 1) * gap
        start  = 0.5 - total / 2
        self.cls_axes = []
        for i, cls in enumerate(self.class_names):
            xpos = start + i * (cls_w + gap)
            ax_c = self.fig.add_axes([xpos, btn_y + 0.055, cls_w, btn_h])
            self.cls_axes.append(ax_c)

        # Style helpers
        def style_btn(ax, label, color="#334466", text_color="white"):
            btn = Button(ax, label,
                         color=color,
                         hovercolor="#556688")
            btn.label.set_color(text_color)
            btn.label.set_fontsize(10)
            return btn

        self.btn_prev = style_btn(ax_prev, "◀  Prev")
        self.btn_next = style_btn(ax_next, "Next  ▶")

        self.cls_buttons = []
        for i, cls in enumerate(self.class_names):
            col = CLASS_COLORS.get(cls, "#334466")
            btn = style_btn(self.cls_axes[i], cls, color=col)
            btn.label.set_fontweight("bold")
            self.cls_buttons.append(btn)

        # Counter text
        self.counter_ax = self.fig.add_axes([0.0, 0.0, 1.0, 0.025])
        self.counter_ax.axis("off")
        self.counter_ax.set_facecolor("#1a1a2e")
        self.counter_text = self.counter_ax.text(
            0.5, 0.5, "",
            ha="center", va="center",
            fontsize=9, color="#aaaacc",
            transform=self.counter_ax.transAxes
        )

        # Wire up callbacks
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        for i, btn in enumerate(self.cls_buttons):
            btn.on_clicked(lambda _, i=i: self._on_class(i))

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ----------------------------------------------------------

    def _draw(self):
        cls   = self.current_class
        idx   = self.current_idx
        ep    = self.epochs_per_class[cls]
        times = ep.times
        data  = ep.get_data()[idx] * 1e6   # V → µV  shape: (n_ch, n_times)
        color = CLASS_COLORS.get(cls, "#aaaaff")

        # Update title
        self.title_text.set_text(
            f"Class: {cls}   |   Epoch {idx + 1} / {self._n_epochs()}"
        )
        self.title_text.set_color(color)

        # Draw each channel
        ch_names = ep.ch_names
        for i, ax in enumerate(self.ch_axes):
            ax.cla()
            ax.set_facecolor("#0f0f23")
            for spine in ax.spines.values():
                spine.set_color("#444466")
            ax.tick_params(colors="#aaaacc", labelsize=7)

            if i >= len(ch_names):
                ax.set_visible(False)
                continue

            signal = data[i]
            ax.plot(times, signal, color=color, linewidth=1.2)
            ax.axvline(0, color="#666688", linewidth=0.8, linestyle="--")
            ax.axhline(0, color="#444466", linewidth=0.5)
            ax.set_title(ch_names[i], fontsize=9, color="#ccccee", pad=3)
            ax.set_xlabel("Time (s)", fontsize=7, color="#888899")
            if i % 4 == 0:
                ax.set_ylabel("µV", fontsize=7, color="#888899")
            ax.grid(True, alpha=0.2, color="#333355")

            # Y-axis symmetric around 0
            ylim = max(abs(signal.min()), abs(signal.max())) * 1.15 or 1
            ax.set_ylim(-ylim, ylim)

        # Highlight active class button
        for i, btn in enumerate(self.cls_buttons):
            active = self.class_names[i] == cls
            fc = CLASS_COLORS.get(self.class_names[i], "#334466")
            btn.ax.set_facecolor(fc if active else "#222244")
            btn.label.set_alpha(1.0 if active else 0.5)

        # Counter
        all_counts = "  |  ".join(
            f"{c}: {self._n_epochs(c)}" for c in self.class_names
        )
        self.counter_text.set_text(f"Epoch counts  →  {all_counts}")

        self.fig.canvas.draw_idle()

    # ----------------------------------------------------------  callbacks

    def _on_prev(self, _=None):
        if self.current_idx > 0:
            self.current_idx -= 1
        else:
            # Wrap to last epoch of previous class
            ci = self.class_names.index(self.current_class)
            if ci > 0:
                self.current_class = self.class_names[ci - 1]
                self.current_idx   = self._n_epochs() - 1
        self._draw()

    def _on_next(self, _=None):
        if self.current_idx < self._n_epochs() - 1:
            self.current_idx += 1
        else:
            # Wrap to first epoch of next class
            ci = self.class_names.index(self.current_class)
            if ci < len(self.class_names) - 1:
                self.current_class = self.class_names[ci + 1]
                self.current_idx   = 0
        self._draw()

    def _on_class(self, class_idx):
        cls = self.class_names[class_idx]
        if cls != self.current_class:
            self.current_class = cls
            self.current_idx   = 0
            self._draw()

    def _on_key(self, event):
        if event.key in ("right", "n", " "):
            self._on_next()
        elif event.key in ("left", "p", "b"):
            self._on_prev()
        elif event.key.isdigit():
            i = int(event.key) - 1
            if 0 <= i < len(self.class_names):
                self._on_class(i)


# ============================================================
# MAIN
# ============================================================

def main(session_path):
    print(f"\nLoading session: {session_path}")

    eeg_df, markers, metadata = load_session(session_path)
    raw, eeg_start_unix, sfreq = build_raw(eeg_df, metadata)

    print(f"  EEG: {len(eeg_df)} samples @ {sfreq} Hz")
    print(f"  Channels: {CHANNEL_NAMES[:len(metadata['eeg_channels'])]}")
    print(f"\n  Filtering {L_FREQ}–{H_FREQ} Hz...")
    raw_f = filter_raw(raw)

    print("\n  Extracting epochs...")
    epochs_per_class = build_epochs_per_class(
        raw_f, markers, sfreq, eeg_start_unix, metadata
    )

    if not epochs_per_class:
        print("No epochs found — check your session folder.")
        return

    print("\nOpening browser... (← → arrows or Prev/Next buttons to navigate)")
    EpochBrowser(epochs_per_class)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        session_path = sys.argv[1]
    else:
        try:
            sessions = sorted([
                os.path.join("data", d)
                for d in os.listdir("data")
                if os.path.isdir(os.path.join("data", d))
                and os.path.exists(os.path.join("data", d, "eeg_raw.csv"))
            ])
            if not sessions:
                raise FileNotFoundError
            session_path = sessions[-1]
            print(f"Using most recent session: {session_path}")
        except FileNotFoundError:
            print("No session found in data/")
            print("Usage: python browse_epochs.py data/my_session_folder")
            sys.exit(1)

    main(session_path)
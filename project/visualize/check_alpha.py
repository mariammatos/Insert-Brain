# ============================================================
# FILE: check_alpha.py
#
# Verifica pico alpha (10 Hz) em olhos abertos vs fechados.
# Usa os primeiros 60s da sessão (30s open + 30s closed).
#
# What to look for: on the eyes-closed curve (orange) you should
# see a clear bump peaking around 10 Hz that's absent or smaller
# on the eyes-open curve (blue). If both lines look identical,
# electrode contact is poor. If eyes-closed shows the bump on
# posterior channels (CP3, CP4, P3, P4) but not frontal ones
# (FC3, FC4), that's completely normal and actually a good sign —
# alpha is strongest over occipital/parietal areas.
#
# Uso: python check_alpha.py <session_path>
# ============================================================

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

SESSION_PATH = r"C:\Users\afons\Projetos\NEURO\Insert-Brain\data"  # fallback

SFREQ        = 250
BASELINE_SEC = 30   # duração de cada bloco
L_FREQ, H_FREQ = 8.0, 30.0

CHANNEL_NAMES = ["FCz", "P3", "CP4", "CP3", "P4", "C3", "FC4", "FC3"]

BG = "#0F0F14"

# ============================================================

def bandpass(data, lo, hi, fs):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(6, [lo/( fs/2), hi/(fs/2)], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=1)

def load_and_split(session_path):
    eeg_df  = pd.read_csv(os.path.join(session_path, "eeg_raw.csv"))
    markers = pd.read_csv(os.path.join(session_path, "markers.csv"))

    ch_cols = [c for c in eeg_df.columns if c.startswith("ch_")]
    eeg     = eeg_df[ch_cols].values.T.astype(np.float64)
    ts      = eeg_df["timestamp"].values

    def get_block(event_name):
        row = markers[markers["event"] == event_name]
        if row.empty:
            raise ValueError(f"Marker '{event_name}' não encontrado.")
        t_start = float(row.iloc[0]["timestamp"])
        i_start = int((t_start - ts[0]) * SFREQ)
        i_end   = i_start + BASELINE_SEC * SFREQ
        i_end   = min(i_end, eeg.shape[1])
        return eeg[:, i_start:i_end]

    open_eeg   = get_block("baseline_open_start")
    closed_eeg = get_block("baseline_closed_start")
    return open_eeg, closed_eeg

def plot_alpha(open_eeg, closed_eeg):
    n_ch = min(open_eeg.shape[0], len(CHANNEL_NAMES))

    open_f   = bandpass(open_eeg,   L_FREQ, H_FREQ, SFREQ)
    closed_f = bandpass(closed_eeg, L_FREQ, H_FREQ, SFREQ)

    ncols = 4
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Alpha check — olhos abertos vs fechados", color="#CCC", fontsize=13)

    axes_flat = axes.flatten() if n_ch > 1 else [axes]

    for i in range(n_ch):
        ax = axes_flat[i]
        ax.set_facecolor(BG)

        fo, po = welch(open_f[i],   fs=SFREQ, nperseg=4*SFREQ)
        fc, pc = welch(closed_f[i], fs=SFREQ, nperseg=4*SFREQ)

        mask = (fo >= 1) & (fo <= 40)
        ax.semilogy(fo[mask], po[mask], color="#4C9BE8", linewidth=1.2, label="Olhos abertos")
        ax.semilogy(fc[mask], pc[mask], color="#E8774C", linewidth=1.2, label="Olhos fechados")

        # Alpha band shading
        ax.axvspan(8, 12, alpha=0.12, color="#FFDD77")
        ax.axvline(10, color="#FFDD77", linewidth=0.6, alpha=0.5)

        ax.set_title(CHANNEL_NAMES[i], color="#AAA", fontsize=9)
        ax.tick_params(colors="#555", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#333")
        ax.set_xlabel("Hz", color="#666", fontsize=7)

        if i == 0:
            ax.legend(fontsize=7, framealpha=0.2, facecolor="#222",
                      edgecolor="#444", labelcolor="#CCC")

    for j in range(n_ch, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()

# ============================================================

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else SESSION_PATH
    open_eeg, closed_eeg = load_and_split(path)
    print(f"Olhos abertos:  {open_eeg.shape[1]/SFREQ:.1f}s")
    print(f"Olhos fechados: {closed_eeg.shape[1]/SFREQ:.1f}s")
    plot_alpha(open_eeg, closed_eeg)

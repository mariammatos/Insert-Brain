# ============================================================
# FILE: visualize_session.py
#
# Uso: python visualize_session.py <session_path>
# Ex:  python visualize_session.py data/P001_20250521_143000
#
# Mostra o sinal EEG com os markers sobrepostos.
# ============================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ============================================================
# CONFIGURAÇÃO VISUAL
# ============================================================

LABEL_COLORS = {
    0: "#888888",   # REST  — cinzento
    1: "#4C9BE8",   # LEFT  — azul
    2: "#E8774C",   # RIGHT — laranja
    3: "#4CE87A",   # FEET  — verde
}

LABEL_NAMES = {
    0: "REST",
    1: "LEFT",
    2: "RIGHT",
    3: "FEET",
}

# Canais a mostrar (índices) — mostra os primeiros 8
N_CHANNELS_PLOT = 8


# ============================================================
# LOAD
# ============================================================

def load_session(session_path):

    eeg_df  = pd.read_csv(os.path.join(session_path, "eeg_raw.csv"))
    markers = pd.read_csv(os.path.join(session_path, "markers.csv"))

    with open(os.path.join(session_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    return eeg_df, markers, metadata


# ============================================================
# PLOT
# ============================================================

def plot_session(session_path):

    print(f"\nA carregar sessão: {session_path}")

    eeg_df, markers, metadata = load_session(session_path)

    sfreq      = metadata["sampling_rate"]
    n_ch       = len(metadata["eeg_channels"])
    n_plot     = min(N_CHANNELS_PLOT, n_ch)

    # Colunas EEG
    eeg_cols   = [f"ch_{i}" for i in range(n_ch)]
    eeg_data   = eeg_df[eeg_cols].values.T          # (n_ch, n_samples)
    timestamps = eeg_df["timestamp"].values          # Unix time

    # Converte timestamps para segundos relativos ao início
    t = timestamps - timestamps[0]

    n_samples  = eeg_data.shape[1]
    duration   = n_samples / sfreq

    print(f"  {n_ch} canais | {n_samples} amostras | {duration:.1f}s | {sfreq}Hz")
    print(f"  {len(markers)} markers")

    # Converte timestamps dos markers para tempo relativo
    markers = markers.copy()
    markers["t_rel"] = markers["timestamp"] - timestamps[0]

    # --------------------------------------------------------
    # Figura
    # --------------------------------------------------------

    fig = plt.figure(figsize=(18, 2.5 * n_plot + 2))
    fig.patch.set_facecolor("#0F0F14")

    gs = GridSpec(
        n_plot + 1, 1,
        figure=fig,
        hspace=0.08,
        height_ratios=[0.4] + [1] * n_plot
    )

    # --------------------------------------------------------
    # Painel superior: linha do tempo dos markers
    # --------------------------------------------------------

    ax_timeline = fig.add_subplot(gs[0])
    ax_timeline.set_facecolor("#0F0F14")
    ax_timeline.set_xlim(0, duration)
    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_yticks([])
    ax_timeline.tick_params(colors="#555", labelsize=8)
    ax_timeline.spines["top"].set_visible(False)
    ax_timeline.spines["right"].set_visible(False)
    ax_timeline.spines["left"].set_visible(False)
    ax_timeline.spines["bottom"].set_color("#333")
    ax_timeline.set_title(
        f"Sessão: {os.path.basename(session_path)}  |  "
        f"{duration:.0f}s  |  {sfreq:.0f}Hz  |  {n_ch}ch",
        color="#CCCCCC", fontsize=11, pad=8, loc="left"
    )

    for _, row in markers.iterrows():
        label  = int(row["label"])
        color  = LABEL_COLORS.get(label, "#FFFFFF")
        t_mark = row["t_rel"]

        if 0 <= t_mark <= duration:
            ax_timeline.axvline(t_mark, color=color, alpha=0.8, linewidth=1.2)
            ax_timeline.text(
                t_mark, 0.55,
                LABEL_NAMES.get(label, str(label)),
                color=color, fontsize=6.5, ha="center", va="bottom",
                rotation=90
            )

    # Legenda
    patches = [
        mpatches.Patch(color=c, label=LABEL_NAMES[l])
        for l, c in LABEL_COLORS.items()
    ]
    ax_timeline.legend(
        handles=patches,
        loc="upper right",
        fontsize=8,
        framealpha=0.2,
        facecolor="#222",
        edgecolor="#444",
        labelcolor="#CCC"
    )

    # --------------------------------------------------------
    # Canais EEG
    # --------------------------------------------------------

    channel_names = [f"ch_{i}" for i in range(n_plot)]

    # Tenta nomes 10-20 se tiver 8 ou 16 canais
    names_8  = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"]
    names_16 = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "P3", "P4"
    ]

    if n_ch == 8:
        channel_names = names_8[:n_plot]
    elif n_ch == 16:
        channel_names = names_16[:n_plot]

    axes = []
    for i in range(n_plot):
        ax = fig.add_subplot(gs[i + 1], sharex=ax_timeline if i > 0 else None)
        axes.append(ax)

        ch_data = eeg_data[i]

        # Normaliza para visualização (z-score robusto)
        med    = np.median(ch_data)
        mad    = np.median(np.abs(ch_data - med)) + 1e-6
        ch_z   = (ch_data - med) / mad
        ch_z   = np.clip(ch_z, -5, 5)

        ax.plot(t, ch_z, color="#00D4AA", linewidth=0.4, alpha=0.85)
        ax.set_facecolor("#0F0F14")
        ax.set_ylim(-6, 6)
        ax.set_yticks([])
        ax.set_xlim(0, duration)

        ax.text(
            -0.01, 0.5, channel_names[i],
            transform=ax.transAxes,
            color="#AAAAAA", fontsize=9,
            ha="right", va="center"
        )

        # Linhas verticais dos markers
        for _, row in markers.iterrows():
            label  = int(row["label"])
            color  = LABEL_COLORS.get(label, "#FFFFFF")
            t_mark = row["t_rel"]
            if 0 <= t_mark <= duration:
                ax.axvline(t_mark, color=color, alpha=0.25, linewidth=0.8)

        for spine in ax.spines.values():
            spine.set_visible(False)

        if i < n_plot - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Tempo (s)", color="#888", fontsize=9)
            ax.tick_params(colors="#555", labelsize=8)
            ax.spines["bottom"].set_visible(True)
            ax.spines["bottom"].set_color("#333")

    plt.tight_layout()

    # Guarda figura
    out_path = os.path.join(session_path, "eeg_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0F0F14")
    print(f"\n  Figura guardada: {out_path}")

    plt.show()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        # Se não der argumento, usa a sessão mais recente
        sessions = sorted([
            os.path.join("data", d)
            for d in os.listdir("data")
            if os.path.isdir(os.path.join("data", d))
        ])

        if not sessions:
            print("Nenhuma sessão encontrada em data/")
            sys.exit(1)

        session_path = sessions[-1]
        print(f"A usar sessão mais recente: {session_path}")
    else:
        session_path = sys.argv[1]

    plot_session(session_path)
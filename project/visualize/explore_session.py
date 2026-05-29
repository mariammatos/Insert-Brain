# ============================================================
# FILE: explore_session.py
#
# Análise exploratória de uma sessão EEG antes de treinar.
# Serve para verificar se há sinal distinguível entre classes
# antes de investir em mais aquisições.
#
# Gera 4 figuras:
#   1. Espectros médios por classe (C3, Cz, C4)
#   2. ERD/ERS time-frequency por classe
#   3. Topografias de potência por banda e por classe
#   4. CSP patterns aprendidos (LEFT vs RIGHT, HANDS vs FEET)
#
# Uso:
#   python explore_session.py
#   python explore_session.py data/maria4_1779448943.511749
# ============================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne
from mne.preprocessing import ICA
from mne.decoding import CSP
from mne.time_frequency import tfr_multitaper

mne.set_log_level("WARNING")


# ============================================================
# PARÂMETROS — devem ser iguais ao train_subject_model.py
# ============================================================

L_FREQ     = 8.0
H_FREQ     = 30.0
EPOCH_TMIN = 0.5
EPOCH_TMAX = 4.5    # janela larga para ver o ERD desenvolver-se
N_CSP      = 4
RANDOM_SEED = 42

#CHANNEL_NAMES = ["FCz", "Cz", "CP4", "CP3", "C4", "C3", "FC4", "FC3"]
CHANNEL_NAMES = ["FCz", "P3", "CP4", "CP3", "P4", "C3", "FC4", "FC3"]

CLASS_COLORS = {
    "REST":  "#888888",
    "LEFT":  "#e74c3c",
    "RIGHT": "#3498db",
    "FEET":  "#2ecc71",
}


# ============================================================
# LOAD + BUILD RAW  (igual ao train_subject_model.py)
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
    ch_types = ["eeg"] * n_ch

    eeg_cols = [c for c in eeg_df.columns if c.startswith("ch_")]
    data     = eeg_df[eeg_cols].values.T * 1e-6   # µV → V

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
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


# ============================================================
# BUILD EPOCHS PER CLASS
# ============================================================

def build_epochs_per_class(raw_f, markers, sfreq, eeg_start_unix, metadata):
    """
    Devolve um dict {class_name: mne.Epochs} usando apenas markers
    do tipo 'mi_start' — o onset real da imagética, não o cue.
    Se não houver 'mi_start', cai back para todos os markers com label > 0.
    """

    classes = metadata.get("classes", {})

    label_to_name = {
        v.get("label"): k
        for k, v in classes.items()
    }

    # Usa mi_start se existir, senão usa cue_on
    if "mi_start" in markers["event"].values:
        cue_markers = markers[markers["event"] == "mi_start"].copy()
        print("  A usar markers 'mi_start' como onset da imagética.")
    else:
        cue_markers = markers[markers["label"] >= 0].copy()
        # Remove duplicados por label+timestamp próximo (cue_on/mi_start/mi_end)
        cue_markers = cue_markers.drop_duplicates(subset=["label"])
        print("  'mi_start' não encontrado — a usar primeiro marker por label.")

    # Remove labels desconhecidos e baseline (-1)
    cue_markers = cue_markers[cue_markers["label"].isin(label_to_name.keys())]
    cue_markers = cue_markers[cue_markers["label"] >= 0]

    if len(cue_markers) == 0:
        raise ValueError("Nenhum marker de imagética encontrado.")

    # Constrói eventos MNE
    events      = []
    bad_markers = []

    for _, row in cue_markers.iterrows():
        t_rel  = row["timestamp"] - eeg_start_unix
        sample = int(round(t_rel * sfreq))

        if t_rel < 0 or sample >= raw_f.n_times:
            bad_markers.append(row["label"])
            continue

        events.append([sample, 0, int(row["label"])])

    if bad_markers:
        print(f"  ⚠ {len(bad_markers)} marker(s) fora do registo ignorados.")

    events     = np.array(events, dtype=int)
    event_map  = {name: info["label"] for name, info in classes.items() if info["label"] >= 0}
    event_map_str = {k: v for k, v in event_map.items()}

    epochs = mne.Epochs(
        raw_f, events,
        event_id=event_map_str,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=None, preload=True, verbose=False
    )

    print(f"\n  Epochs por classe:")
    epochs_per_class = {}
    for name, label in event_map.items():
        try:
            ep = epochs[name]
            print(f"    {name:>8}: {len(ep)} epochs")
            epochs_per_class[name] = ep
        except KeyError:
            print(f"    {name:>8}: 0 epochs (sem dados)")

    return epochs_per_class


# ============================================================
# FIGURA 1: ESPECTROS POR CLASSE
# ============================================================

def plot_spectra(epochs_per_class, sfreq, session_path):
    """
    PSD média por classe nos canais C3, Cz, C4.
    Mostra onde as classes se separam espectralmente.
    """

    #target_chs = ["C3", "Cz", "C4"]
    target_chs = CHANNEL_NAMES

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
    fig.suptitle("Espectros médios por classe", fontsize=13, fontweight="bold")

    for i, ch_name in enumerate(target_chs):
        ax = axes[i // 4, i % 4]

        for class_name, ep in epochs_per_class.items():
            if ch_name not in ep.ch_names:
                continue

            ch_idx = ep.ch_names.index(ch_name)
            data   = ep.get_data()[:, ch_idx, :]   # (n_epochs, n_times)

            # Welch PSD
            from scipy.signal import welch
            freqs_all, psds = zip(*[
                welch(trial, fs=sfreq, nperseg=min(256, data.shape[1]))
                for trial in data
            ])

            freqs = freqs_all[0]
            psd_mean = np.mean(psds, axis=0)
            psd_std  = np.std(psds, axis=0)

            # Só banda 4-40 Hz
            mask = (freqs >= 4) & (freqs <= 40)
            f    = freqs[mask]
            m    = psd_mean[mask]
            s    = psd_std[mask]

            color = CLASS_COLORS.get(class_name, "black")
            ax.semilogy(f, m, label=class_name, color=color, linewidth=2)
            ax.fill_between(f, m - s, m + s, alpha=0.15, color=color)

        # Bandas de referência
        ax.axvspan(8,  12, alpha=0.08, color="orange", label="mu (8-12Hz)")
        ax.axvspan(13, 30, alpha=0.08, color="blue",   label="beta (13-30Hz)")

        ax.set_title(ch_name, fontsize=11)
        ax.set_xlabel("Frequência (Hz)")
        ax.set_ylabel("PSD (V²/Hz)" if i % 4 == 0 else "")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(session_path, "explore_spectra.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Espectros: {path}")


# ============================================================
# FIGURA 2: ERD/ERS TIME-FREQUENCY
# ============================================================

def plot_erd(epochs_per_class, session_path):
    """
    Time-frequency (multitaper) médio por classe em C3 e C4.
    O ERD aparece como descida de potência (azul) após o onset.
    """

    target_chs = ["C3", "C4"]
    freqs      = np.arange(6, 35, 1)
    n_cycles   = freqs / 2.0

    n_classes = len(epochs_per_class)
    n_chs     = len(target_chs)

    fig, axes = plt.subplots(
        n_classes, n_chs,
        figsize=(n_chs * 5, n_classes * 3),
        squeeze=False
    )
    fig.suptitle(
        "Time-Frequency (ERD/ERS) — onset = 0s\n"
        "Azul = descida de potência (ERD = bom sinal de MI)",
        fontsize=11, fontweight="bold"
    )

    im = None

    for row, (class_name, ep) in enumerate(epochs_per_class.items()):

        for col, ch_name in enumerate(target_chs):

            ax = axes[row][col]

            if ch_name not in ep.ch_names or len(ep) == 0:
                ax.set_visible(False)
                continue

            try:
                # .copy() evita que pick_channels modifique o epochs original
                power = tfr_multitaper(
                    ep.copy().pick_channels([ch_name], ordered=False),
                    freqs=freqs,
                    n_cycles=n_cycles,
                    time_bandwidth=2.0,
                    return_itc=False,
                    verbose=False
                )

                # Baseline: média dos primeiros 0.5s após EPOCH_TMIN
                baseline_mask  = (power.times >= EPOCH_TMIN) & (power.times < EPOCH_TMIN + 0.5)
                baseline_power = power.data[0, :, baseline_mask].mean(axis=-1, keepdims=True)
                erd            = 10 * np.log10(power.data[0] / (baseline_power + 1e-30))

                im = ax.imshow(
                    erd,
                    aspect="auto",
                    origin="lower",
                    extent=[power.times[0], power.times[-1], freqs[0], freqs[-1]],
                    vmin=-3, vmax=3,
                    cmap="RdBu_r"
                )
                ax.axvline(0, color="white", linewidth=1.5, linestyle="--")
                ax.axhline(12, color="white", linewidth=0.8, linestyle=":")
                ax.axhline(8,  color="white", linewidth=0.8, linestyle=":")

            except Exception as e:
                ax.text(0.5, 0.5, f"Erro:\n{e}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)

            ax.set_title(f"{class_name} — {ch_name}", fontsize=9)
            ax.set_xlabel("Tempo (s)")
            ax.set_ylabel("Freq (Hz)" if col == 0 else "")

    if im is not None:
        plt.colorbar(im, ax=axes, label="ERD (dB)", shrink=0.6)
    plt.tight_layout()
    path = os.path.join(session_path, "explore_erd.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ ERD/ERS:  {path}")


# ============================================================
# FIGURA 3: TOPOGRAFIAS DE POTÊNCIA
# ============================================================

def plot_topos(epochs_per_class, session_path):
    """
    Topografia da potência média nas bandas mu e beta por classe.
    Deve mostrar activação assimétrica em C3/C4 para LEFT/RIGHT.
    """

    bands = {
        "mu (8-12Hz)":   (8,  12),
        "beta (13-30Hz)":(13, 30),
    }

    n_classes = len(epochs_per_class)
    n_bands   = len(bands)

    fig, axes = plt.subplots(
        n_classes, n_bands,
        figsize=(n_bands * 3, n_classes * 2.5),
        squeeze=False
    )
    fig.suptitle("Topografias de potência por classe e banda",
                 fontsize=11, fontweight="bold")

    from scipy.signal import welch

    for row, (class_name, ep) in enumerate(epochs_per_class.items()):

        for col, (band_name, (fmin, fmax)) in enumerate(bands.items()):

            ax = axes[row][col]

            if len(ep) == 0:
                ax.set_visible(False)
                continue

            data  = ep.get_data()           # (n_epochs, n_ch, n_times)
            sfreq = ep.info["sfreq"]
            topo  = np.zeros(data.shape[1])

            for ch_idx in range(data.shape[1]):
                psds = []
                for trial in data[:, ch_idx, :]:
                    f, psd = welch(trial, fs=sfreq, nperseg=min(256, trial.shape[0]))
                    mask   = (f >= fmin) & (f <= fmax)
                    psds.append(psd[mask].mean())
                topo[ch_idx] = np.mean(psds)

            try:
                mne.viz.plot_topomap(
                    topo, ep.info,
                    axes=ax, show=False,
                    cmap="RdYlBu_r",
                    vlim=(np.percentile(topo, 10), np.percentile(topo, 90))
                )
            except Exception as e:
                ax.text(0.5, 0.5, str(e), ha="center", va="center",
                        transform=ax.transAxes, fontsize=6)

            if row == 0:
                ax.set_title(band_name, fontsize=9)
            if col == 0:
                ax.set_ylabel(class_name, fontsize=9)

    plt.tight_layout()
    path = os.path.join(session_path, "explore_topos.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Topografias: {path}")


# ============================================================
# FIGURA 4: CSP PATTERNS
# ============================================================

def plot_csp_patterns(epochs_per_class, session_path):
    """
    Treina CSP para cada par de classes e mostra os patterns.
    Se os patterns mostrarem actividade centromótora (C3/C4/Cz)
    é bom sinal — o CSP encontrou algo real.
    Se mostrarem eléctrodos de borda (FC3, FCz) é provável artefacto.
    """

    pairs = [
        ("LEFT",  "RIGHT", "Direction: LEFT vs RIGHT"),
        ("LEFT",  "FEET",  "Axis: HANDS vs FEET (LEFT vs FEET)"),
        ("RIGHT", "FEET",  "Axis: HANDS vs FEET (RIGHT vs FEET)"),
    ]

    valid_pairs = [
        (a, b, title) for a, b, title in pairs
        if a in epochs_per_class and b in epochs_per_class
        and len(epochs_per_class[a]) >= 2 and len(epochs_per_class[b]) >= 2
    ]

    if not valid_pairs:
        print("  ⚠ Sem pares de classes com dados suficientes para CSP.")
        return

    n_pairs  = len(valid_pairs)
    n_comps  = min(N_CSP, 4)   # mostra no máximo 4 componentes

    fig, axes = plt.subplots(
        n_pairs, n_comps * 2,
        figsize=(n_comps * 2 * 2.5, n_pairs * 2.5),
        squeeze=False
    )
    fig.suptitle(
        "CSP Patterns — actividade centromótora (C3/C4/Cz) é bom sinal",
        fontsize=11, fontweight="bold"
    )

    for row, (cls_a, cls_b, title) in enumerate(valid_pairs):

        ep_a = epochs_per_class[cls_a]
        ep_b = epochs_per_class[cls_b]

        X = np.concatenate([ep_a.get_data(), ep_b.get_data()], axis=0)
        y = np.array([0] * len(ep_a) + [1] * len(ep_b))

        try:
            csp = CSP(n_components=n_comps, reg="ledoit_wolf", log=True)
            csp.fit(X, y)

            # patterns_ são os patterns espaciais (n_components, n_channels)
            patterns = csp.patterns_[:n_comps * 2]   # primeiros e últimos

            info_plot = ep_a.info

            for col in range(n_comps * 2):

                ax = axes[row][col]

                if col >= len(patterns):
                    ax.set_visible(False)
                    continue

                try:
                    mne.viz.plot_topomap(
                        patterns[col], info_plot,
                        axes=ax, show=False,
                        cmap="RdBu_r"
                    )
                except Exception:
                    ax.set_visible(False)

                ax.set_title(f"CSP {col+1}", fontsize=8)

            # Label da linha
            axes[row][0].set_ylabel(title, fontsize=8)

        except Exception as e:
            axes[row][0].text(
                0.5, 0.5, f"CSP falhou:\n{e}",
                ha="center", va="center",
                transform=axes[row][0].transAxes, fontsize=7
            )

    plt.tight_layout()
    path = os.path.join(session_path, "explore_csp.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ CSP patterns: {path}")


# ============================================================
# FIGURA 5: SINAL RAW POR CANAL E CLASSE (sanity check)
# ============================================================

def plot_raw_epochs(epochs_per_class, session_path):
    """
    Média de epochs por classe em C3, Cz, C4.
    Não deve ter artefactos óbvios (picos, drift, saturação).
    Se a média de epochs tiver forma de onda suave é bom sinal.
    """

    target_chs = ["C3", "Cz", "C4"]

    fig, axes = plt.subplots(1, len(target_chs), figsize=(15, 4), sharey=False)
    fig.suptitle(
        "Média de epochs por classe (sinal filtrado 8-30Hz)\n"
        "Verifica artefactos: picos, drift, saturação",
        fontsize=11, fontweight="bold"
    )

    for ax, ch_name in zip(axes, target_chs):

        for class_name, ep in epochs_per_class.items():

            if ch_name not in ep.ch_names or len(ep) == 0:
                continue

            ch_idx = ep.ch_names.index(ch_name)
            data   = ep.get_data()[:, ch_idx, :] * 1e6   # V → µV
            times  = ep.times

            mean = data.mean(axis=0)
            std  = data.std(axis=0)

            color = CLASS_COLORS.get(class_name, "black")
            ax.plot(times, mean, label=class_name, color=color, linewidth=2)
            ax.fill_between(times, mean - std, mean + std, alpha=0.15, color=color)

        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.set_title(ch_name, fontsize=11)
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Amplitude (µV)" if ch_name == target_chs[0] else "")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(session_path, "explore_raw_epochs.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Raw epochs: {path}")


# ============================================================
# MAIN
# ============================================================

def explore_session(session_path):

    print("\n" + "=" * 60)
    print("ANÁLISE EXPLORATÓRIA DA SESSÃO EEG")
    print(f"Sessão: {session_path}")
    print("=" * 60)

    print("\n[1/3] A carregar e pré-processar dados...")
    eeg_df, markers, metadata = load_session(session_path)
    raw, eeg_start_unix, sfreq = build_raw(eeg_df, metadata)
    raw_f = filter_raw(raw)

    print(f"  EEG: {len(eeg_df)} amostras @ {sfreq} Hz")
    print(f"  Canais: {CHANNEL_NAMES[:len(metadata['eeg_channels'])]}")
    print(f"\n  Markers disponíveis:")
    print(markers.groupby(["event", "label"]).size().to_string())

    print("\n[2/3] A extrair epochs por classe...")
    epochs_per_class = build_epochs_per_class(
        raw_f, markers, sfreq, eeg_start_unix, metadata
    )

    print("\n[3/3] A gerar figuras...")
    plot_spectra(epochs_per_class, sfreq, session_path)
    plot_erd(epochs_per_class, session_path)
    plot_topos(epochs_per_class, session_path)
    plot_csp_patterns(epochs_per_class, session_path)
    plot_raw_epochs(epochs_per_class, session_path)

    print("\n" + "=" * 60)
    print("ANÁLISE CONCLUÍDA")
    print(f"Figuras guardadas em: {session_path}")
    print("=" * 60)

    print("""
O QUE PROCURAR NAS FIGURAS:

  explore_spectra.png
    → As curvas de LEFT/RIGHT devem ser diferentes de FEET em mu e beta
    → Se estão todas sobrepostas: sem sinal distinguível

  explore_erd.png
    → Deve aparecer azul (ERD = descida de potência) após t=0
    → Em C3 para LEFT, em C4 para RIGHT, em Cz para FEET
    → Se não há azul: o sujeito não está a fazer imagética motora

  explore_topos.png
    → LEFT deve ter mais actividade em C4, RIGHT em C3 (contralateral)
    → FEET deve ter activação central (Cz)
    → Se as topografias estão todas iguais: sem diferença espacial

  explore_csp.png
    → Os patterns devem ter actividade em C3/C4/Cz
    → Se a actividade está nos canais de borda (FC3, FCz, CP3):
      provável artefacto muscular ou de movimento

  explore_raw_epochs.png
    → Verifica se há picos, drift, ou saturação
    → Sinal limpo deve ser suave e centrado em 0
""")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        session_path = sys.argv[1]
    else:
        # Usa sessão mais recente automaticamente
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
            print(f"A usar sessão mais recente: {session_path}")
        except FileNotFoundError:
            print("Nenhuma sessão encontrada em data/")
            print("Uso: python explore_session.py data/P001_20250521_143000")
            sys.exit(1)

    explore_session(session_path)
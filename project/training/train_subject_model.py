# ============================================================
# FILE: training/train_subject_model.py
# ============================================================
#
# Pipeline de treino específico por sujeito.
# Corre após a aquisição e guarda os modelos na pasta da sessão.
#
# Três classificadores em cascata:
#
#   [1] GATING:    REST (0) vs ACTIVE (1)
#         ↓ se ACTIVE
#   [2] AXIS:      Mãos (0) vs Pés (1)
#         ↓ se Mãos
#   [3] DIRECTION: Esquerda (1) vs Direita (2)
#
# No robô: só avança para o próximo nível se o anterior disser ACTIVE/Mãos.
#
# Inputs esperados na pasta da sessão:
#   eeg_raw.csv   → sinal EEG + coluna "timestamp"
#   markers.csv   → timestamp, event, label
#   metadata.json → sampling_rate, eeg_channels, classes, timings
#
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne.preprocessing import ICA
from mne.decoding import CSP

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    cohen_kappa_score,
    ConfusionMatrixDisplay,
    classification_report,
    balanced_accuracy_score,
    f1_score,
)

mne.set_log_level("WARNING")


# ============================================================
# PARÂMETROS
# ============================================================

L_FREQ      = 8.0
H_FREQ      = 30.0

# Janela de extração relativa ao onset do cue (segundos).
# EPOCH_TMIN / EPOCH_TMAX são usados APENAS para definir as Epochs do MNE.
# A âncora temporal (início do EEG) é sempre calculada a partir do CSV.
# Janela relativa ao início da motor imagery
EPOCH_TMIN = 0.5
EPOCH_TMAX = 4.5

N_CSP       = 4
RANDOM_SEED = 42
N_FOLDS     = 5


# ============================================================
# LOAD DATA
# ============================================================

def load_session_data(session_path):
    """
    Lê EEG, markers e metadata de uma pasta de sessão.

    Returns
    -------
    eeg_df    : DataFrame com colunas ch_0..ch_N + timestamp
    markers   : DataFrame com colunas timestamp, event, label
    metadata  : dict com sampling_rate, eeg_channels, classes, timings
    """
    eeg_df  = pd.read_csv(os.path.join(session_path, "eeg_raw.csv"))
    markers = pd.read_csv(os.path.join(session_path, "markers.csv"))

    with open(os.path.join(session_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    # Validações básicas de integridade
    if "timestamp" not in eeg_df.columns:
        raise ValueError("eeg_raw.csv não tem coluna 'timestamp'.")
    if "timestamp" not in markers.columns:
        raise ValueError("markers.csv não tem coluna 'timestamp'.")

    eeg_cols = [c for c in eeg_df.columns if c.startswith("ch_")]
    if len(eeg_cols) == 0:
        raise ValueError("eeg_raw.csv não tem colunas 'ch_*'.")

    return eeg_df, markers, metadata


# ============================================================
# BUILD MNE RAW
# ============================================================

def build_mne_raw(eeg_df, metadata):
    """
    Constrói um mne.RawArray a partir do CSV da sessão.

    O timestamp Unix da primeira amostra do EEG é definido como
    meas_date do Raw para que a referência temporal fique registada
    (embora o alinhamento com markers seja feito explicitamente em
    build_epochs, sem depender deste campo).

    Returns
    -------
    raw            : mne.RawArray já com montagem definida
    eeg_start_unix : float — timestamp Unix da 1ª amostra do EEG
    """
    sfreq = metadata["sampling_rate"]

    # Nomes dos canais OpenBCI Cyton+Daisy (8 canais)
    #CHANNEL_NAMES = ["FCz", "Cz", "CP4", "CP3", "C4", "C3", "FC4", "FC3"]
    CHANNEL_NAMES = ["FCz", "P3", "CP4", "CP3", "P4", "C3", "FC4", "FC3"]


    n_ch     = len(metadata["eeg_channels"])
    ch_names = CHANNEL_NAMES[:n_ch]
    ch_types = ["eeg"] * n_ch

    eeg_cols = [c for c in eeg_df.columns if c.startswith("ch_")]
    data     = eeg_df[eeg_cols].values.T * 1e-6  # µV → V

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw  = mne.io.RawArray(data, info, verbose=False)

    # Guardar o timestamp da 1ª amostra como meas_date
    # (informativo; o alinhamento real é feito em build_epochs)
    eeg_start_unix = float(eeg_df["timestamp"].iloc[0])
    raw.set_meas_date(eeg_start_unix)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_eeg_reference("average", verbose=False)
    raw.set_montage(montage, on_missing="ignore", verbose=False)

    return raw, eeg_start_unix


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_raw(raw):

    raw.filter(L_FREQ, H_FREQ,
               fir_design="firwin",
               verbose=False)
    raw.notch_filter(freqs=[25, 50], verbose=False) 

    return raw


# ============================================================
# BUILD EPOCHS FROM MARKERS
# ============================================================

def build_epochs(raw, markers, sfreq, label_filter, eeg_start_unix):
    """
    Cria mne.Epochs a partir dos markers do DataLogger.

    O alinhamento é feito usando eeg_start_unix (timestamp Unix da
    primeira amostra do EEG), garantindo que os markers são mapeados
    para a posição correcta no sinal independentemente de quando a
    aquisição começou.

    Parameters
    ----------
    raw            : mne.RawArray pré-processado
    markers        : DataFrame com colunas timestamp, event, label
    sfreq          : float — frequência de amostragem (Hz)
    label_filter   : dict {label_int: class_id_for_model}
                     ex: {0: 0, 1: 1, 2: 1, 3: 1}  → REST(0) vs ACTIVE(1)
                     ex: {1: 0, 2: 0, 3: 1}         → HANDS(0) vs FEET(1)
                     ex: {1: 1, 2: 2}               → LEFT(1) vs RIGHT(2)
    eeg_start_unix : float — timestamp Unix da 1ª amostra do EEG
                     (obtido de eeg_df["timestamp"].iloc[0])

    Returns
    -------
    X : np.ndarray (n_epochs, n_channels, n_times)
    y : np.ndarray (n_epochs,)
    """
    cue_markers = markers[
        (markers["event"] == "mi_start") &
        (markers["label"].isin(label_filter.keys()))
    ].copy()

    if len(cue_markers) == 0:
        raise ValueError(
            f"Nenhum marker encontrado para labels {list(label_filter.keys())}."
        )

    # ------------------------------------------------------------------
    # ALINHAMENTO CORRECTO:
    # O índice de amostra de cada marker é calculado relativamente ao
    # início do EEG (primeira amostra do CSV), NÃO relativamente ao
    # primeiro marker.
    # ------------------------------------------------------------------
    events      = []
    bad_markers = []

    for _, row in cue_markers.iterrows():
        # Posição em segundos desde o início do EEG
        t_rel  = row["timestamp"] - eeg_start_unix
        sample = int(round(t_rel * sfreq))

        # Verificações de integridade
        if t_rel < 0:
            bad_markers.append(
                f"  ⚠ Marker (label={row['label']}) com timestamp "
                f"ANTES do início do EEG (t_rel={t_rel:.3f}s). Ignorado."
            )
            continue

        if sample >= raw.n_times:
            bad_markers.append(
                f"  ⚠ Marker (label={row['label']}) em t_rel={t_rel:.3f}s "
                f"(amostra {sample}) fora do registo EEG "
                f"({raw.n_times} amostras). Ignorado."
            )
            continue

        event_id = int(label_filter[row["label"]])
        events.append([sample, 0, event_id])

    if bad_markers:
        print("\n".join(bad_markers))

    if len(events) == 0:
        raise ValueError(
            "Nenhum marker válido encontrado após verificação de alinhamento. "
            "Verifica os timestamps do EEG e dos markers."
        )

    events = np.array(events, dtype=int)

    # Aviso se a perda de markers for elevada
    n_total = len(cue_markers)
    n_valid = len(events)
    if n_valid < n_total:
        pct_lost = (n_total - n_valid) / n_total * 100
        print(
            f"  ⚠ {n_total - n_valid}/{n_total} markers ignorados "
            f"({pct_lost:.1f}%). Verifica a sincronização EEG/markers."
        )

    event_id_map = {str(v): v for v in set(label_filter.values())}

    epochs = mne.Epochs(
        raw, events, event_id=event_id_map,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=None, preload=True, verbose=False
    )

    n_dropped = n_valid - len(epochs)           # já tens esta linha mais abaixo — move-a para cá
    if n_dropped > 0:
        print(f"  ⚠ {n_dropped} epoch(s) rejeitados (artefacto de amplitude).")

    # Verificar se algum epoch foi descartado pelo MNE (fora dos limites)
    n_dropped = n_valid - len(epochs)
    if n_dropped > 0:
        print(
            f"  ⚠ {n_dropped} epoch(s) descartados pelo MNE "
            f"(provavelmente próximos do fim do registo)."
        )

    return epochs.get_data(), epochs.events[:, -1]


# ============================================================
# TRAIN + EVALUATE
# ============================================================

def train_pipeline(X, y, classifier_name):
    """
    Treina CSP → Scaler → LDA com Stratified K-Fold CV.
    Devolve o modelo final treinado em todos os dados + métricas.
    """
    print(f"\n  [{classifier_name}] {len(X)} épocas | classes: {np.unique(y)}")

    clf = Pipeline([
        ("CSP",    CSP(n_components=N_CSP, reg="ledoit_wolf", log=True)),
        ("Scaler", StandardScaler()),
        ("LDA",    LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto"
        ))
    ])

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    # ----------------------------------------------------------
    # Cross-validation
    # ----------------------------------------------------------

    acc_scores = cross_val_score(
        clf, X, y, cv=cv, scoring="accuracy"
    )

    kappa_scores = cross_val_score(
        clf, X, y, cv=cv,
        scoring=lambda est, Xt, yt: cohen_kappa_score(yt, est.predict(Xt))
    )

    y_pred_cv = cross_val_predict(clf, X, y, cv=cv)

    # ----------------------------------------------------------
    # Métricas adicionais
    # ----------------------------------------------------------

    balanced_acc = balanced_accuracy_score(y, y_pred_cv)
    f1           = f1_score(y, y_pred_cv, average="weighted")
    report       = classification_report(y, y_pred_cv, output_dict=True)

    # ----------------------------------------------------------
    # Print resultados
    # ----------------------------------------------------------

    print(
        f"  Accuracy:          {acc_scores.mean()*100:.1f}% "
        f"± {acc_scores.std()*100:.1f}%"
    )
    print(
        f"  Kappa:             {kappa_scores.mean():.3f} "
        f"± {kappa_scores.std():.3f}"
    )
    print(f"  Balanced Accuracy: {balanced_acc*100:.1f}%")
    print(f"  Weighted F1-Score: {f1:.3f}")

    # ----------------------------------------------------------
    # Avaliação automática da qualidade
    # ----------------------------------------------------------

    mean_acc   = acc_scores.mean()
    mean_kappa = kappa_scores.mean()

    print("\n  Avaliação do modelo:")
    if mean_acc >= 0.75 and mean_kappa >= 0.4:
        print("   ✓ Modelo com boa qualidade.")
        print("   ✓ Aquisição provavelmente suficiente.")
    elif mean_acc >= 0.65:
        print("   ⚠ Modelo utilizável mas instável.")
        print("   ⚠ Recomenda-se repetir aquisição se possível.")
    else:
        print("   ✗ Performance fraca.")
        print("   ✗ Repetir aquisição recomendado.")

    # ----------------------------------------------------------
    # Treino final em TODOS os dados
    # ----------------------------------------------------------

    clf.fit(X, y)

    return clf, {
        "acc_mean":              float(acc_scores.mean()),
        "acc_std":               float(acc_scores.std()),
        "kappa_mean":            float(kappa_scores.mean()),
        "kappa_std":             float(kappa_scores.std()),
        "balanced_accuracy":     float(balanced_acc),
        "f1_score":              float(f1),
        "classification_report": report,
        "y_true":                y,
        "y_pred":                y_pred_cv,
    }


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

def save_confusion_matrices(results, session_path):
    """
    Gera e guarda uma figura com as 3 confusion matrices em linha.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    configs = [
        ("gating",    ["REST",     "ACTIVE"],  "Reds",   "GATING: Rest vs Active"),
        ("axis",      ["Mãos",     "Pés"],     "Blues",  "AXIS: Mãos vs Pés"),
        ("direction", ["Esquerda", "Direita"], "Greens", "DIRECTION: Esq vs Dir"),
    ]

    for ax, (key, labels, cmap, title) in zip(axes, configs):
        ConfusionMatrixDisplay.from_predictions(
            y_true=results[key]["y_true"],
            y_pred=results[key]["y_pred"],
            display_labels=labels,
            cmap=cmap,
            ax=ax,
            colorbar=False
        )
        ax.set_title(title)

    plt.tight_layout()

    path = os.path.join(session_path, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Confusion matrices guardadas: {path}")


# ============================================================
# MAIN
# ============================================================

def train_subject_model(session_path):
    """
    Pipeline completo de treino para um sujeito.
    Recebe o caminho da pasta da sessão e guarda lá os três modelos.
    """
    print("\n" + "=" * 60)
    print("TRAINING SUBJECT-SPECIFIC MODEL")
    print(f"Session: {session_path}")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Load
    # ----------------------------------------------------------

    print("\n[1/5] A carregar dados da sessão...")

    eeg_df, markers, metadata = load_session_data(session_path)
    sfreq = metadata["sampling_rate"]

    print(f"  EEG: {len(eeg_df)} amostras @ {sfreq} Hz")
    print(f"  Markers: {len(markers)} eventos")

    # ----------------------------------------------------------
    # 2. Build MNE Raw
    #
    # eeg_start_unix é o timestamp Unix da 1ª amostra do EEG.
    # É a âncora temporal para todos os cálculos de alinhamento.
    # ----------------------------------------------------------

    print("\n[2/5] A construir objeto MNE Raw...")

    raw, eeg_start_unix = build_mne_raw(eeg_df, metadata)

    print(f"  EEG start (Unix): {eeg_start_unix:.3f} s")
    print(f"  Markers range:    {markers['timestamp'].min():.3f} "
          f"– {markers['timestamp'].max():.3f} s")

    # Verificação de sanidade: todos os markers devem estar dentro do registo
    eeg_end_unix = eeg_start_unix + (len(eeg_df) - 1) / sfreq
    markers_before = (markers["timestamp"] < eeg_start_unix).sum()
    markers_after  = (markers["timestamp"] > eeg_end_unix).sum()

    if markers_before > 0:
        print(
            f"  ⚠ ATENÇÃO: {markers_before} marker(s) com timestamp "
            f"ANTES do início do EEG! Verifica a sincronização."
        )
    if markers_after > 0:
        print(
            f"  ⚠ ATENÇÃO: {markers_after} marker(s) com timestamp "
            f"DEPOIS do fim do EEG! Verifica a sincronização."
        )

    # ----------------------------------------------------------
    # 3. Preprocess
    # ----------------------------------------------------------

    print("\n[3/5] A pré-processar (filtro bandpass)...")

    raw = preprocess_raw(raw)

    # ----------------------------------------------------------
    # 4. Extrair labels do config e treinar classificadores
    # ----------------------------------------------------------

    print("\n[4/5] A extrair épocas e treinar classificadores...")

    classes = metadata.get("classes", {})

    rest_label  = classes.get("REST",  {}).get("label", 0)
    left_label  = classes.get("LEFT",  {}).get("label", 1)
    right_label = classes.get("RIGHT", {}).get("label", 2)
    feet_label  = classes.get("FEET",  {}).get("label", 3)

    # ----------------------------------------------------------
    # [1] GATING: REST (0) vs ACTIVE (1)
    # ----------------------------------------------------------

    gating_filter = {
        rest_label:  0,
        left_label:  1,
        right_label: 1,
        feet_label:  1,
    }

    X_gate, y_gate = build_epochs(
        raw, markers, sfreq, gating_filter, eeg_start_unix
    )
    clf_gate, gate_metrics = train_pipeline(X_gate, y_gate, "GATING (Rest vs Active)")

    # ----------------------------------------------------------
    # [2] AXIS: Mãos/LEFT+RIGHT (0) vs Pés/FEET (1)
    # ----------------------------------------------------------

    axis_filter = {
        left_label:  0,
        right_label: 0,
        feet_label:  1,
    }

    X_axis, y_axis = build_epochs(
        raw, markers, sfreq, axis_filter, eeg_start_unix
    )
    clf_axis, axis_metrics = train_pipeline(X_axis, y_axis, "AXIS (Mãos vs Pés)")

    # ----------------------------------------------------------
    # [3] DIRECTION: Esquerda (1) vs Direita (2)
    # ----------------------------------------------------------

    dir_filter = {
        left_label:  left_label,
        right_label: right_label,
    }

    X_dir, y_dir = build_epochs(
        raw, markers, sfreq, dir_filter, eeg_start_unix
    )
    clf_dir, dir_metrics = train_pipeline(X_dir, y_dir, "DIRECTION (Esq vs Dir)")

    # ----------------------------------------------------------
    # 5. Guardar modelos + relatório
    # ----------------------------------------------------------

    print("\n[5/5] A guardar modelos e relatório...")

    paths = {
        "gating":    os.path.join(session_path, "model_gating.pkl"),
        "axis":      os.path.join(session_path, "model_axis.pkl"),
        "direction": os.path.join(session_path, "model_direction.pkl"),
    }

    joblib.dump(clf_gate, paths["gating"])
    joblib.dump(clf_axis, paths["axis"])
    joblib.dump(clf_dir,  paths["direction"])

    for name, path in paths.items():
        print(f"  Modelo {name.upper():>10} guardado: {path}")

    # Relatório JSON
    report = {
        "session_path": session_path,
        "alignment": {
            "eeg_start_unix": eeg_start_unix,
            "eeg_end_unix":   eeg_end_unix,
            "markers_min":    float(markers["timestamp"].min()),
            "markers_max":    float(markers["timestamp"].max()),
        },
        "gating": {
            "acc_mean":          gate_metrics["acc_mean"],
            "acc_std":           gate_metrics["acc_std"],
            "kappa_mean":        gate_metrics["kappa_mean"],
            "kappa_std":         gate_metrics["kappa_std"],
            "balanced_accuracy": gate_metrics["balanced_accuracy"],
            "f1_score":          gate_metrics["f1_score"],
        },
        "axis": {
            "acc_mean":          axis_metrics["acc_mean"],
            "acc_std":           axis_metrics["acc_std"],
            "kappa_mean":        axis_metrics["kappa_mean"],
            "kappa_std":         axis_metrics["kappa_std"],
            "balanced_accuracy": axis_metrics["balanced_accuracy"],
            "f1_score":          axis_metrics["f1_score"],
        },
        "direction": {
            "acc_mean":          dir_metrics["acc_mean"],
            "acc_std":           dir_metrics["acc_std"],
            "kappa_mean":        dir_metrics["kappa_mean"],
            "kappa_std":         dir_metrics["kappa_std"],
            "balanced_accuracy": dir_metrics["balanced_accuracy"],
            "f1_score":          dir_metrics["f1_score"],
        },
        "params": {
            "n_csp":      N_CSP,
            "l_freq":     L_FREQ,
            "h_freq":     H_FREQ,
            "epoch_tmin": EPOCH_TMIN,
            "epoch_tmax": EPOCH_TMAX,
            "n_folds":    N_FOLDS,
        }
    }

    report_path = os.path.join(session_path, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"  Relatório guardado: {report_path}")

    # Confusion matrices (as 3 juntas)
    save_confusion_matrices(
        {"gating": gate_metrics, "axis": axis_metrics, "direction": dir_metrics},
        session_path
    )

    # ----------------------------------------------------------
    # Sumário final
    # ----------------------------------------------------------

    print("\n" + "=" * 60)
    print("TREINO CONCLUÍDO")
    print("=" * 60)
    print(
        f"  GATING    → "
        f"{gate_metrics['acc_mean']*100:.1f}% "
        f"± {gate_metrics['acc_std']*100:.1f}%  |  "
        f"κ = {gate_metrics['kappa_mean']:.3f}"
    )
    print(
        f"  AXIS      → "
        f"{axis_metrics['acc_mean']*100:.1f}% "
        f"± {axis_metrics['acc_std']*100:.1f}%  |  "
        f"κ = {axis_metrics['kappa_mean']:.3f}"
    )
    print(
        f"  DIRECTION → "
        f"{dir_metrics['acc_mean']*100:.1f}% "
        f"± {dir_metrics['acc_std']*100:.1f}%  |  "
        f"κ = {dir_metrics['kappa_mean']:.3f}"
    )
    print("=" * 60 + "\n")

    return clf_gate, clf_axis, clf_dir, report


# ============================================================
# ENTRY POINT (uso direto, fora do main.py)
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Uso: python train_subject_model.py <session_path>")
        print("Ex:  python train_subject_model.py data/P001_20250521_143000")
        sys.exit(1)

    train_subject_model(sys.argv[1])
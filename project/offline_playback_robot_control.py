# ============================================================
# FILE: offline_playback_robot_control.py
#
# BCI playback system with PyQt5 graphical interface.
# EEG preprocessing and classification match evaluate_classifier.
#
# Robot behavior:
#   LEFT / RIGHT → continuous command repeat at MOVE_INTERVAL
#   REST         → stops movement
#   FEET         → switches active servo and stops movement
#
# Epoch extraction is always based on real dataset timestamps
# (t_mi_start + EPOCH_TMIN/TMAX), independent of robot actions.
#
# Usage:
#   python playback_robot_gui.py <session_path> [--no-robot] [--speed N]
#   python playback_robot_gui.py data/S020_demo --no-robot
# ============================================================

import sys
import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import mne
from mne.preprocessing import ICA

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    from config import CONFIG
    _DEFAULT_SFREQ = CONFIG.get("sfreq", 160)
except Exception:
    _DEFAULT_SFREQ = 160

from PyQt5.QtCore    import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt5.QtGui     import QFont, QColor, QPainter, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QFrame, QSizePolicy, QComboBox, QPushButton
)


# ============================================================
# PARÂMETROS
# ============================================================

EPOCH_TMIN    = 1.0
EPOCH_TMAX    = 3.5

# Intervalo entre comandos repetidos durante movimento contínuo (LEFT/RIGHT)
# Mais baixo = mais rápido. 0.6s é suave mas perceptível.
MOVE_INTERVAL = 0.6   # segundos

L_FREQ        = 8.0
H_FREQ        = 30.0
NOTCH_FREQ    = 50.0
RANDOM_SEED   = 42

ROBOT_BAUD    = 9600
COMMAND_MAP   = {0: None, 1: "LEFT", 2: "RIGHT", 3: "FEET"}
NAMES         = {0: "REST", 1: "LEFT", 2: "RIGHT", 3: "FEET"}
SYMBOLS       = {0: "■",   1: "←",   2: "→",     3: "↓"}
N_SERVOS      = 3
SERVO_JOINTS  = ["Base", "Shoulder", "Elbow"]

BG        = "#0d0d14"
BG2       = "#13131f"
BG3       = "#1a1a2e"
ACCENT    = "#00d4aa"
LEFT_COL  = "#4c9be8"
RIGHT_COL = "#e87a4c"
FEET_COL  = "#a64ce8"
REST_COL  = "#666688"
STOP_COL  = "#e84c4c"
TEXT      = "#e0e0f0"
TEXT_DIM  = "#444466"
BORDER    = "#2a2a40"
GREEN     = "#00d4aa"
RED_COL   = "#e84c4c"
YELLOW    = "#e8c84c"

CLASS_COLOR = {0: REST_COL, 1: LEFT_COL, 2: RIGHT_COL, 3: FEET_COL}


# ============================================================
# PRÉ-PROCESSAMENTO MNE — idêntico ao evaluate_classifier
# ============================================================

def preprocess_mne(eeg, sfreq, ch_names):
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw  = mne.io.RawArray(eeg, info, verbose=False)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore", verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=NOTCH_FREQ, method="fir", verbose=False)
    n_components = min(len(ch_names) - 1, 7)
    ica = ICA(n_components=n_components, random_state=RANDOM_SEED,
              method="fastica", max_iter=500)
    ica.fit(raw, verbose=False)
    muscle_idx, _ = ica.find_bads_muscle(raw, verbose=False)
    if muscle_idx:
        ica.exclude = muscle_idx
        ica.apply(raw, verbose=False)
    return raw


# ============================================================
# CARREGAR DADOS
# ============================================================

def load_session(session_path):
    for fname in ("eeg_raw.csv", "eeg_data.csv"):
        p = os.path.join(session_path, fname)
        if os.path.exists(p):
            eeg_path = p; break
    else:
        raise FileNotFoundError(f"Nenhum ficheiro EEG em '{session_path}'")
    df      = pd.read_csv(eeg_path).sort_values("timestamp").reset_index(drop=True)
    ch_cols = [c for c in df.columns if c.startswith("ch_")]
    eeg     = df[ch_cols].values.T
    times   = df["timestamp"].values
    markers = pd.read_csv(os.path.join(session_path, "markers.csv"))
    sfreq_path = os.path.join(session_path, "sfreq.txt")
    if os.path.exists(sfreq_path):
        sfreq = float(open(sfreq_path).read().strip())
    else:
        dt = np.diff(times[:min(500, len(times))])
        sfreq = float(1.0 / np.median(dt[dt > 0]))
    return eeg, times, markers, sfreq, ch_cols

def load_models(session_path):
    keys = ("gating", "axis", "direction")
    missing = [k for k in keys
               if not os.path.exists(os.path.join(session_path, f"model_{k}.pkl"))]
    if missing:
        raise FileNotFoundError(f"Modelos não encontrados: {missing}")
    return tuple(joblib.load(os.path.join(session_path, f"model_{k}.pkl")) for k in keys)


# ============================================================
# CLASSIFICAÇÃO — idêntica ao evaluate_classifier
# ============================================================

def classify(epoch, clf_gate, clf_axis, clf_dir):
    w = epoch[np.newaxis, :, :]
    if clf_gate.predict(w)[0] == 0:
        return 0
    feet_label = clf_axis.classes_[1]
    if clf_axis.predict(w)[0] == feet_label:
        return 3
    pred_dir   = clf_dir.predict(w)[0]
    left_label = clf_dir.classes_[0]
    return 1 if pred_dir == left_label else 2


# ============================================================
# SERIAL HELPERS
# ============================================================

def send_command(rob, cmd):
    if not rob or not cmd:
        return
    try:
        rob.write((cmd + "\n").encode())
        time.sleep(0.02)
    except Exception:
        pass

def close_robot(rob):
    if rob and rob.is_open:
        rob.close()


# ============================================================
# PLAYBACK WORKER
# ============================================================

class PlaybackWorker(QObject):
    status_update = pyqtSignal(str, str)          # msg, level
    event_update  = pyqtSignal(int, int, int, int, bool)  # idx, total, gt, pred, correct
    servo_update  = pyqtSignal(int)
    phase_update  = pyqtSignal(str, int)          # phase, label
    stats_update  = pyqtSignal(int, int)          # n_correct, n_wrong
    log_signal    = pyqtSignal(str)
    finished      = pyqtSignal()
    preprocessed  = pyqtSignal()                  # preprocessing done

    def __init__(self, session_path, speed):
        super().__init__()
        self.session_path = session_path
        self.speed        = speed
        self._stop        = False
        self.rob          = None     # set externally by GUI after connect

    def stop(self):
        self._stop = True

    def run(self):
        sp = self.speed if self.speed > 0 else 99999.0

        self.log_signal.emit("A carregar modelos...")
        clf_gate, clf_axis, clf_dir = load_models(self.session_path)
        self.log_signal.emit(
            f"Modelos: gate{list(clf_gate.classes_)} "
            f"axis{list(clf_axis.classes_)} "
            f"dir{list(clf_dir.classes_)}"
        )

        eeg, times, markers, sfreq, ch_cols = load_session(self.session_path)
        self.log_signal.emit(
            f"EEG: {eeg.shape[0]}ch × {eeg.shape[1]}s @ {sfreq:.0f}Hz  "
            f"({times[-1]:.1f}s total)"
        )

        self.log_signal.emit("Pré-processamento MNE (avg ref → bandpass → notch → ICA)...")
        self.status_update.emit("Pré-processamento…", "info")
        raw      = preprocess_mne(eeg, sfreq, ch_cols)
        eeg_proc = raw.get_data()
        self.log_signal.emit("EEG pré-processado.")
        self.status_update.emit("Pronto — a iniciar playback", "ok")
        self.preprocessed.emit()

        cues   = markers[markers["event"] == "cue_on"].sort_values("timestamp").reset_index(drop=True)
        starts = markers[markers["event"] == "mi_start"].sort_values("timestamp").reset_index(drop=True)
        ends   = markers[markers["event"] == "mi_end"].sort_values("timestamp").reset_index(drop=True)
        n_events = min(len(cues), len(starts), len(ends))
        self.log_signal.emit(f"{n_events} épocas de MI identificadas.")

        servo_idx   = 0
        n_correct   = 0
        n_wrong     = 0
        active_cmd  = None   # LEFT or RIGHT currently being repeated

        for i in range(n_events):
            if self._stop:
                break

            t_cue      = float(cues.iloc[i]["timestamp"])
            t_mi_start = float(starts.iloc[i]["timestamp"])
            t_mi_end   = float(ends.iloc[i]["timestamp"])
            gt_label   = int(starts.iloc[i]["label"])

            # ── CUE phase ──────────────────────────────────────────
            # During cue, keep repeating any active directional command
            self.phase_update.emit("cue", gt_label)
            cue_dur  = t_mi_start - t_cue
            deadline = time.time() + cue_dur / sp
            while time.time() < deadline and not self._stop:
                if active_cmd in ("LEFT", "RIGHT"):
                    send_command(self.rob, active_cmd)
                    self.phase_update.emit("moving", NAMES_TO_LABEL.get(active_cmd, 0))
                remaining = deadline - time.time()
                time.sleep(min(MOVE_INTERVAL / sp, max(remaining, 0)))

            if self._stop:
                break

            # ── Settling ───────────────────────────────────────────
            self.phase_update.emit("settling", gt_label)
            settle_deadline = time.time() + EPOCH_TMIN / sp
            while time.time() < settle_deadline and not self._stop:
                if active_cmd in ("LEFT", "RIGHT"):
                    send_command(self.rob, active_cmd)
                remaining = settle_deadline - time.time()
                time.sleep(min(MOVE_INTERVAL / sp, max(remaining, 0)))

            # ── Extract epoch & classify ───────────────────────────
            # (timing uses real dataset timestamps — unaffected by robot behaviour)
            t_win_start = t_mi_start + EPOCH_TMIN
            t_win_end   = t_mi_start + EPOCH_TMAX
            mask  = (times >= t_win_start) & (times <= t_win_end)
            epoch = eeg_proc[:, mask] if mask.sum() >= 2 else None

            if epoch is None:
                self.log_signal.emit(f"Época {i+1}: fora dos limites — saltado")
                self.phase_update.emit("skip", 0)
                time.sleep(1.0 / sp)
                continue

            pred    = classify(epoch, clf_gate, clf_axis, clf_dir)
            cmd     = COMMAND_MAP.get(pred)
            correct = (pred == gt_label)

            if correct:
                n_correct += 1
            else:
                n_wrong += 1

            # ── Apply command & update state ───────────────────────
            if pred == 0:
                # REST: stop continuous movement
                active_cmd = None

            elif pred == 3:
                # FEET: switch servo, stop movement
                active_cmd = None
                send_command(self.rob, "FEET")
                servo_idx = (servo_idx + 1) % N_SERVOS

            else:
                # LEFT or RIGHT: send once immediately, then keep repeating
                active_cmd = cmd
                send_command(self.rob, cmd)

            self.servo_update.emit(servo_idx)
            self.event_update.emit(i + 1, n_events, gt_label, pred, correct)
            self.stats_update.emit(n_correct, n_wrong)
            self.phase_update.emit("classify", pred)

            self.log_signal.emit(
                f"[{i+1}/{n_events}]  gt:{NAMES[gt_label]}  "
                f"pred:{NAMES[pred]}  {'✓' if correct else '✗'}  "
                f"{cmd or '—'}  active:{active_cmd or 'none'}"
            )

            # ── Remaining MI time: keep repeating if directional ───
            remaining_mi = t_mi_end - t_win_end
            if remaining_mi > 0:
                mi_deadline = time.time() + remaining_mi / sp
                while time.time() < mi_deadline and not self._stop:
                    if active_cmd in ("LEFT", "RIGHT"):
                        send_command(self.rob, active_cmd)
                        self.phase_update.emit("moving", pred)
                    wait = min(MOVE_INTERVAL / sp, max(mi_deadline - time.time(), 0))
                    time.sleep(wait)

        # End: stop robot
        active_cmd = None
        close_robot(self.rob)
        self.phase_update.emit("done", 0)
        self.finished.emit()


# Label reverse lookup for display during movement
NAMES_TO_LABEL = {"LEFT": 1, "RIGHT": 2, "FEET": 3, "REST": 0}


# ============================================================
# ARM DIAGRAM
# ============================================================

class ArmDiagram(QWidget):
    def __init__(self):
        super().__init__()
        self.active = 0
        self.setMinimumSize(220, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_active(self, idx):
        self.active = idx
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h   = self.width(), self.height()
        cx, cy = w // 2, h // 2 + 30
        p.fillRect(0, 0, w, h, QColor(BG2))

        joints = [
            (cx,      cy),
            (cx + 18, cy - 55),
            (cx + 52, cy - 105),
        ]

        for i in range(len(joints) - 1):
            col = QColor(ACCENT) if i == self.active else QColor(BORDER)
            pen = QPen(col, 6 if i == self.active else 3)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(joints[i][0], joints[i][1], joints[i+1][0], joints[i+1][1])

        for i, (jx, jy) in enumerate(joints):
            is_active = (i == self.active)
            r = 10 if is_active else 6
            if is_active:
                p.setPen(QPen(QColor(ACCENT + "44"), 4))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(jx - r - 5, jy - r - 5, (r+5)*2, (r+5)*2)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor(ACCENT)))
            else:
                p.setPen(QPen(QColor(BORDER), 2))
                p.setBrush(QBrush(QColor(BG3)))
            p.drawEllipse(jx - r, jy - r, r*2, r*2)

            p.setPen(QColor(ACCENT) if is_active else QColor(TEXT_DIM))
            p.setFont(QFont("Courier New", 8, QFont.Bold if is_active else QFont.Normal))
            p.drawText(jx + r + 5, jy + 4, SERVO_JOINTS[i])

        p.end()


# ============================================================
# PREDICTION INDICATOR
# ============================================================

class PredIndicator(QWidget):
    def __init__(self, label, symbol, color, parent=None):
        super().__init__(parent)
        self.label  = label
        self.symbol = symbol
        self.color  = color
        self.active = False
        self.setMinimumHeight(54)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_active(self, val):
        self.active = val
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        col  = QColor(self.color)

        if self.active:
            p.setBrush(QBrush(QColor(self.color + "44")))
            p.setPen(QPen(col, 1.5))
        else:
            p.setBrush(QBrush(QColor(self.color + "0d")))
            p.setPen(QPen(QColor(self.color + "33"), 1))

        p.drawRoundedRect(1, 1, w-2, h-2, 6, 6)

        p.setPen(col if self.active else QColor(self.color + "55"))
        p.setFont(QFont("Courier New", 12, QFont.Bold if self.active else QFont.Normal))
        p.drawText(0, 0, w, h, Qt.AlignCenter, f"{self.symbol}  {self.label}")

        if self.active:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            p.drawEllipse(w - 14, 6, 8, 8)

        p.end()


# ============================================================
# SERIAL READER THREAD (for Arduino responses)
# ============================================================

class SerialReader(QThread):
    line_received = pyqtSignal(str)

    def __init__(self, port):
        super().__init__()
        self.port    = port
        self._active = True

    def run(self):
        while self._active:
            try:
                if self.port and self.port.is_open and self.port.in_waiting:
                    line = self.port.readline().decode(errors="ignore").strip()
                    if line:
                        self.line_received.emit(line)
            except Exception:
                pass
            time.sleep(0.02)

    def stop(self):
        self._active = False


# ============================================================
# MAIN WINDOW
# ============================================================

class PlaybackWindow(QWidget):

    def __init__(self, session_path, speed):
        super().__init__()
        self.session_path  = session_path
        self.speed         = speed
        self._rob          = None
        self._serial_reader = None
        self._worker       = None
        self._thread       = None

        self.setWindowTitle("Insert-Brain — BCI Playback")
        self.resize(720, 680)
        self.setStyleSheet(f"background-color: {BG}; color: {TEXT};")

        self._build_ui()
        self._populate_ports()
        self._start_worker()   # starts preprocessing immediately; robot optional

    # ── Build UI ─────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 14, 16, 14)

        # Header
        title = QLabel("BCI PLAYBACK")
        title.setFont(QFont("Courier New", 15, QFont.Bold))
        title.setStyleSheet(f"color: {ACCENT}; letter-spacing: 4px;")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        sub = QLabel(os.path.basename(self.session_path))
        sub.setFont(QFont("Courier New", 8))
        sub.setStyleSheet(f"color: {TEXT_DIM}; letter-spacing: 1px;")
        sub.setAlignment(Qt.AlignCenter)
        root.addWidget(sub)

        root.addWidget(self._hline())

        # ── Connection row ────────────────────────────────────
        conn_row = QHBoxLayout()

        port_lbl = QLabel("Port:")
        port_lbl.setFont(QFont("Courier New", 9))
        port_lbl.setStyleSheet(f"color: {TEXT_DIM};")
        conn_row.addWidget(port_lbl)

        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(130)
        self.port_combo.setStyleSheet(self._combo_style())
        conn_row.addWidget(self.port_combo)

        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setStyleSheet(self._small_btn_style(ACCENT))
        refresh_btn.clicked.connect(self._populate_ports)
        conn_row.addWidget(refresh_btn)

        conn_row.addSpacing(8)

        self.connect_btn = QPushButton("CONNECT")
        self.connect_btn.setFixedWidth(110)
        self.connect_btn.setStyleSheet(self._action_btn_style(ACCENT))
        self.connect_btn.clicked.connect(self._toggle_connect)
        conn_row.addWidget(self.connect_btn)

        conn_row.addSpacing(10)

        self.conn_status = QLabel("● Disconnected")
        self.conn_status.setFont(QFont("Courier New", 9))
        self.conn_status.setStyleSheet(f"color: {STOP_COL};")
        conn_row.addWidget(self.conn_status)

        conn_row.addStretch()

        self.status_lbl = QLabel("A carregar…")
        self.status_lbl.setFont(QFont("Courier New", 9))
        self.status_lbl.setStyleSheet(f"color: {TEXT_DIM};")
        self.status_lbl.setAlignment(Qt.AlignRight)
        conn_row.addWidget(self.status_lbl)

        root.addLayout(conn_row)
        root.addWidget(self._hline())

        # ── Main body ─────────────────────────────────────────
        body = QHBoxLayout()
        body.setSpacing(14)

        # Left panel: arm + servo + accuracy
        left = QVBoxLayout()
        left.setSpacing(8)

        servo_title = QLabel("ACTIVE SERVO")
        servo_title.setFont(QFont("Courier New", 8))
        servo_title.setStyleSheet(f"color: {TEXT_DIM}; letter-spacing: 2px;")
        servo_title.setAlignment(Qt.AlignCenter)
        left.addWidget(servo_title)

        self.active_lbl = QLabel("── BASE ──")
        self.active_lbl.setFont(QFont("Courier New", 13, QFont.Bold))
        self.active_lbl.setStyleSheet(f"color: {ACCENT}; letter-spacing: 3px;")
        self.active_lbl.setAlignment(Qt.AlignCenter)
        left.addWidget(self.active_lbl)

        self.arm = ArmDiagram()
        left.addWidget(self.arm)

        pill_row = QHBoxLayout()
        pill_row.setSpacing(6)
        self._pills = []
        for i in range(N_SERVOS):
            pill = QLabel(f"S{i+1}")
            pill.setAlignment(Qt.AlignCenter)
            pill.setFixedSize(36, 24)
            pill.setFont(QFont("Courier New", 9, QFont.Bold))
            self._pills.append(pill)
            pill_row.addWidget(pill)
        self._refresh_pills(0)
        left.addLayout(pill_row)

        left.addWidget(self._hline())

        acc_title = QLabel("ACCURACY")
        acc_title.setFont(QFont("Courier New", 8))
        acc_title.setStyleSheet(f"color: {TEXT_DIM}; letter-spacing: 2px;")
        acc_title.setAlignment(Qt.AlignCenter)
        left.addWidget(acc_title)

        self.acc_lbl = QLabel("—")
        self.acc_lbl.setFont(QFont("Courier New", 22, QFont.Bold))
        self.acc_lbl.setStyleSheet(f"color: {ACCENT};")
        self.acc_lbl.setAlignment(Qt.AlignCenter)
        left.addWidget(self.acc_lbl)

        self.acc_detail = QLabel("")
        self.acc_detail.setFont(QFont("Courier New", 9))
        self.acc_detail.setStyleSheet(f"color: {TEXT_DIM};")
        self.acc_detail.setAlignment(Qt.AlignCenter)
        left.addWidget(self.acc_detail)

        left.addStretch()
        body.addLayout(left, 3)

        div = QFrame()
        div.setFrameShape(QFrame.VLine)
        div.setStyleSheet(f"color: {BORDER};")
        body.addWidget(div)

        # Right panel: indicators + gt + phase + event counter
        right = QVBoxLayout()
        right.setSpacing(8)

        cmd_title = QLabel("CLASSIFIER OUTPUT")
        cmd_title.setFont(QFont("Courier New", 8))
        cmd_title.setStyleSheet(f"color: {TEXT_DIM}; letter-spacing: 2px;")
        right.addWidget(cmd_title)

        self.ind_left  = PredIndicator("LEFT HAND",           "←", LEFT_COL)
        self.ind_right = PredIndicator("RIGHT HAND",          "→", RIGHT_COL)
        self.ind_feet  = PredIndicator("FEET — SWITCH SERVO", "↕", FEET_COL)
        self.ind_rest  = PredIndicator("REST",                "■", REST_COL)
        right.addWidget(self.ind_left)
        right.addWidget(self.ind_right)
        right.addWidget(self.ind_feet)
        right.addWidget(self.ind_rest)

        self._indicators = {1: self.ind_left, 2: self.ind_right,
                            3: self.ind_feet, 0: self.ind_rest}

        right.addWidget(self._hline())

        gt_row = QHBoxLayout()
        gt_lbl_hdr = QLabel("Ground truth:")
        gt_lbl_hdr.setFont(QFont("Courier New", 9))
        gt_lbl_hdr.setStyleSheet(f"color: {TEXT_DIM};")
        self.gt_lbl = QLabel("—")
        self.gt_lbl.setFont(QFont("Courier New", 11, QFont.Bold))
        self.gt_lbl.setStyleSheet(f"color: {TEXT};")
        gt_row.addWidget(gt_lbl_hdr)
        gt_row.addWidget(self.gt_lbl)
        gt_row.addStretch()
        self.verdict_lbl = QLabel("")
        self.verdict_lbl.setFont(QFont("Courier New", 11, QFont.Bold))
        self.verdict_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        gt_row.addWidget(self.verdict_lbl)
        right.addLayout(gt_row)

        self.phase_lbl = QLabel("aguarda…")
        self.phase_lbl.setFont(QFont("Courier New", 10))
        self.phase_lbl.setStyleSheet(f"color: {TEXT_DIM}; letter-spacing: 1px;")
        self.phase_lbl.setAlignment(Qt.AlignCenter)
        right.addWidget(self.phase_lbl)

        self.event_lbl = QLabel("")
        self.event_lbl.setFont(QFont("Courier New", 9))
        self.event_lbl.setStyleSheet(f"color: {TEXT_DIM};")
        self.event_lbl.setAlignment(Qt.AlignCenter)
        right.addWidget(self.event_lbl)

        right.addStretch()
        body.addLayout(right, 4)
        root.addLayout(body)

        root.addWidget(self._hline())

        log_title = QLabel("LOG")
        log_title.setFont(QFont("Courier New", 8))
        log_title.setStyleSheet(f"color: {TEXT_DIM}; letter-spacing: 2px;")
        root.addWidget(log_title)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        self.log.setFont(QFont("Courier New", 8))
        self.log.setStyleSheet(
            f"background: {BG2}; color: {ACCENT}; "
            f"border: 1px solid {BORDER}; border-radius: 4px; padding: 4px;"
        )
        root.addWidget(self.log)

    # ── Port management ───────────────────────────────────────

    def _populate_ports(self):
        self.port_combo.clear()
        if HAS_SERIAL:
            ports = [p.device for p in serial.tools.list_ports.comports()]
            for p in ports:
                self.port_combo.addItem(p)
            if not ports:
                self.port_combo.addItem("No ports found")
        else:
            self.port_combo.addItem("pyserial not installed")

    def _toggle_connect(self):
        if self._rob and self._rob.is_open:
            self._disconnect_robot()
        else:
            self._connect_robot()

    def _connect_robot(self):
        if not HAS_SERIAL:
            self._log("pyserial não instalado")
            return
        port = self.port_combo.currentText()
        try:
            rob = serial.Serial(port, ROBOT_BAUD, timeout=1)
            time.sleep(2)
            self._rob = rob
            if self._worker:
                self._worker.rob = rob   # hand reference to running worker
            self._serial_reader = SerialReader(rob)
            self._serial_reader.line_received.connect(lambda l: self._log(f"← {l}"))
            self._serial_reader.start()
            self.conn_status.setText("● Connected")
            self.conn_status.setStyleSheet(f"color: {ACCENT}; font-family: 'Courier New'; font-size: 9px;")
            self.connect_btn.setText("DISCONNECT")
            self._log(f"Robô ligado em {port} @ {ROBOT_BAUD} baud")
        except Exception as e:
            self._log(f"Erro ao ligar: {e}")

    def _disconnect_robot(self):
        if self._serial_reader:
            self._serial_reader.stop()
            self._serial_reader.wait()
            self._serial_reader = None
        if self._worker:
            self._worker.rob = None
        close_robot(self._rob)
        self._rob = None
        self.conn_status.setText("● Disconnected")
        self.conn_status.setStyleSheet(f"color: {STOP_COL}; font-family: 'Courier New'; font-size: 9px;")
        self.connect_btn.setText("CONNECT")
        self._log("Robô desligado.")

    # ── Worker ────────────────────────────────────────────────

    def _start_worker(self):
        self._thread = QThread()
        self._worker = PlaybackWorker(self.session_path, self.speed)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._on_finished)
        self._worker.status_update.connect(self._on_status)
        self._worker.event_update.connect(self._on_event)
        self._worker.servo_update.connect(self._on_servo)
        self._worker.phase_update.connect(self._on_phase)
        self._worker.stats_update.connect(self._on_stats)
        self._worker.log_signal.connect(self._on_log)
        self._thread.start()

    # ── Slots ─────────────────────────────────────────────────

    def _on_status(self, msg, level):
        colors = {"info": TEXT_DIM, "ok": ACCENT, "warn": YELLOW, "err": RED_COL}
        self.status_lbl.setStyleSheet(f"color: {colors.get(level, TEXT_DIM)};")
        self.status_lbl.setText(msg)

    def _on_event(self, ev_idx, n_total, gt, pred, correct):
        self.event_lbl.setText(f"Evento {ev_idx} / {n_total}")
        gt_col = CLASS_COLOR.get(gt, TEXT)
        self.gt_lbl.setText(f"{SYMBOLS[gt]} {NAMES[gt]}")
        self.gt_lbl.setStyleSheet(
            f"color: {gt_col}; font-family: 'Courier New'; font-size: 11px; font-weight: bold;")
        if correct:
            self.verdict_lbl.setText("✓ correcto")
            self.verdict_lbl.setStyleSheet(f"color: {GREEN}; font-family: 'Courier New'; font-size: 11px;")
        elif pred == 0 or gt == 0:
            self.verdict_lbl.setText("~ rest mix")
            self.verdict_lbl.setStyleSheet(f"color: {YELLOW}; font-family: 'Courier New'; font-size: 11px;")
        else:
            self.verdict_lbl.setText(f"✗ esperado {NAMES[gt]}")
            self.verdict_lbl.setStyleSheet(f"color: {RED_COL}; font-family: 'Courier New'; font-size: 11px;")
        for cls, ind in self._indicators.items():
            ind.set_active(cls == pred)

    def _on_servo(self, idx):
        self.active_lbl.setText(f"── {SERVO_JOINTS[idx].upper()} ──")
        self.arm.set_active(idx)
        self._refresh_pills(idx)

    def _on_phase(self, phase, label):
        texts = {
            "cue":      f"▸ CUE  —  {NAMES.get(label,'?')}",
            "settling": "  aguarda settling…",
            "moving":   f"  ▶ a mover  {SYMBOLS.get(label,'?')} {NAMES.get(label,'?')}",
            "classify": f"● CLASSIFICADO  —  {NAMES.get(label,'?')}",
            "skip":     "  ⚠ época saltada",
            "done":     "✓ playback concluído",
        }
        cols = {
            "cue":      TEXT_DIM,
            "settling": TEXT_DIM,
            "moving":   CLASS_COLOR.get(label, ACCENT),
            "classify": CLASS_COLOR.get(label, ACCENT),
            "skip":     YELLOW,
            "done":     ACCENT,
        }
        self.phase_lbl.setText(texts.get(phase, phase))
        self.phase_lbl.setStyleSheet(
            f"color: {cols.get(phase, TEXT_DIM)}; "
            f"font-family: 'Courier New'; font-size: 10px; letter-spacing: 1px;")

        if phase in ("cue", "settling"):
            for ind in self._indicators.values():
                ind.set_active(False)
            col_gt = CLASS_COLOR.get(label, TEXT)
            self.gt_lbl.setText(f"{SYMBOLS.get(label,'?')} {NAMES.get(label,'?')}")
            self.gt_lbl.setStyleSheet(
                f"color: {col_gt}; font-family: 'Courier New'; font-size: 11px; font-weight: bold;")

    def _on_stats(self, n_correct, n_wrong):
        total = n_correct + n_wrong
        if total == 0:
            return
        acc = n_correct / total * 100
        col = ACCENT if acc >= 60 else YELLOW if acc >= 40 else RED_COL
        self.acc_lbl.setText(f"{acc:.0f}%")
        self.acc_lbl.setStyleSheet(
            f"color: {col}; font-family: 'Courier New'; font-size: 22px; font-weight: bold;")
        self.acc_detail.setText(f"✓ {n_correct}  ✗ {n_wrong}  de {total}")

    def _on_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}]  {msg}")

    def _on_finished(self):
        self._on_status("Playback concluído", "ok")

    # ── Helpers ───────────────────────────────────────────────

    def _refresh_pills(self, active):
        for i, pill in enumerate(self._pills):
            if i == active:
                pill.setStyleSheet(
                    f"background: {ACCENT}22; color: {ACCENT}; "
                    f"border: 1px solid {ACCENT}; border-radius: 4px;")
            else:
                pill.setStyleSheet(
                    f"background: {BG2}; color: {TEXT_DIM}; "
                    f"border: 1px solid {BORDER}; border-radius: 4px;")

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}]  {msg}")

    def _hline(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {BORDER};")
        return line

    def _combo_style(self):
        return (
            f"QComboBox {{ background: {BG2}; color: {TEXT}; "
            f"border: 1px solid {BORDER}; border-radius: 4px; "
            f"padding: 4px 8px; font-family: 'Courier New'; font-size: 9px; }}"
            f"QComboBox::drop-down {{ border: none; }}"
            f"QComboBox QAbstractItemView {{ background: {BG2}; color: {TEXT}; "
            f"border: 1px solid {BORDER}; }}"
        )

    def _action_btn_style(self, color):
        return (
            f"QPushButton {{ background: {color}22; color: {color}; "
            f"border: 1px solid {color}66; border-radius: 4px; "
            f"padding: 4px 10px; font-family: 'Courier New'; font-size: 9px; "
            f"letter-spacing: 1px; }}"
            f"QPushButton:hover {{ background: {color}44; border-color: {color}; }}"
        )

    def _small_btn_style(self, color):
        return (
            f"QPushButton {{ background: {color}22; color: {color}; "
            f"border: 1px solid {color}55; border-radius: 4px; font-size: 13px; }}"
            f"QPushButton:hover {{ background: {color}44; }}"
        )

    def closeEvent(self, event):
        if self._worker:
            self._worker.stop()
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
        self._disconnect_robot()
        event.accept()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Playback BCI com GUI — LEFT/RIGHT movem continuamente até REST ou FEET."
    )
    parser.add_argument("session_path", nargs="?", help="Pasta da sessão")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Velocidade de playback (1.0=tempo real, 2.0=2×, 0=máximo)")
    args = parser.parse_args()

    if args.session_path:
        session_path = args.session_path
    else:
        data_dir = "data"
        sessions = []
        if os.path.isdir(data_dir):
            sessions = sorted([
                os.path.join(data_dir, d)
                for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
                and os.path.exists(os.path.join(data_dir, d, "model_gating.pkl"))
                and any(os.path.exists(os.path.join(data_dir, d, f))
                        for f in ("eeg_raw.csv", "eeg_data.csv"))
            ])
        if not sessions:
            print("Uso: python playback_robot_gui.py <session_path> [--speed N]")
            sys.exit(1)
        session_path = sessions[-1]
        print(f"A usar sessão mais recente: {session_path}")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = PlaybackWindow(session_path, args.speed)
    win.show()
    sys.exit(app.exec_())
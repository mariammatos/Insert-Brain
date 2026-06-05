# ============================================================
# FILE: realtime_robot_control.py
#
# Real-time BCI robot control GUI. Streams EEG via BrainFlow,
# runs the cascade classifier (gating → axis → direction), and
# sends serial commands to the robot arm.
#
# Usage:
#   python realtime_robot_control.py <session_path>
#   python realtime_robot_control.py data/P001_20250521_143000
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

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

from brainflow.board_shim import BoardShim, BrainFlowInputParams
from config import CONFIG

from PyQt5.QtCore    import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt5.QtGui     import QFont, QColor, QPainter, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QFrame, QSizePolicy, QComboBox, QPushButton
)

# Configurações globais do sistema, tempos e aspeto visual da GUI
REST_DURATION  = 3.0    
MI_DURATION    = 5.0    
MOVE_DURATION  = 1.0    

EPOCH_TMIN     = 0.5    
EPOCH_TMAX     = 4.5    

L_FREQ         = 8.0
H_FREQ         = 30.0
NOTCH_FREQS    = [25.0, 50.0]

MOVE_INTERVAL  = 0.25   
ROBOT_BAUD     = 9600

COMMAND_MAP    = {0: None, 1: "LEFT", 2: "RIGHT", 3: "FEET"}
NAMES          = {0: "REST", 1: "LEFT", 2: "RIGHT", 3: "FEET"}
SYMBOLS        = {0: "■",   1: "←",   2: "→",     3: "↓"}

N_SERVOS       = 3
SERVO_JOINTS   = ["Base", "Shoulder", "Elbow"]

BG_COLOR       = "#0d0d14"
BG_PANEL       = "#13131f"
BG_INNER       = "#1a1a2e"
COLOR_ACCENT   = "#00d4aa"
COLOR_LEFT     = "#4c9be8"
COLOR_RIGHT    = "#e87a4c"
COLOR_FEET     = "#a64ce8"
COLOR_REST     = "#666688"
COLOR_STOP     = "#e84c4c"
COLOR_TEXT     = "#e0e0f0"
COLOR_DIM      = "#444466"
COLOR_BORDER   = "#2a2a40"
COLOR_GREEN    = "#00d4aa"
COLOR_RED      = "#e84c4c"
COLOR_YELLOW   = "#e8c84c"

CLASS_COLORS = {0: COLOR_REST, 1: COLOR_LEFT, 2: COLOR_RIGHT, 3: COLOR_FEET}


# Nomes anatómicos dos canais — idêntico ao train_subject_model.py
CHANNEL_NAMES = ["F3", "F4", "FC4", "C3", "FCz", "CP3", "CP4", "CPz"]

# Janela de classificação = EPOCH_TMAX - EPOCH_TMIN do treino (4.5 - 0.5 = 4.0 s)
WINDOW_SEC = EPOCH_TMAX - EPOCH_TMIN   # 4.0 s


# Funções de processamento de sinal e conversão de dados do buffer para épocas MNE
def preprocess_raw(raw):
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=NOTCH_FREQS, method="fir", verbose=False)
    return raw

def process_buffer_to_epoch(eeg_data, sfreq):
    """
    Convert a raw EEG buffer (n_ch, n_samples) into a preprocessed epoch
    ready for clf.predict(), applying the same pipeline used at training:
      1. µV → V
      2. Average reference
      3. FIR bandpass 8-30 Hz
      4. Notch at 25 Hz and 50 Hz
      5. Take the last WINDOW_SEC samples (no fixed onset)
    """
    n_ch      = eeg_data.shape[0]
    n_need    = int(round(sfreq * WINDOW_SEC))
    ch_names  = CHANNEL_NAMES[:n_ch]

    # [1] µV → V (igual ao treino: data * 1e-6)
    eeg_v = eeg_data * 1e-6

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw  = mne.io.RawArray(eeg_v, info, verbose=False)

    # [2] Montage + Average reference (igual ao treino)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore", verbose=False)
    raw.set_eeg_reference("average", verbose=False)

    # [3] FIR bandpass + [4] Notch (igual ao treino)
    raw = preprocess_raw(raw)

    # [5] Pega os últimos n_need samples (janela de classificação sem onset fixo)
    data = raw.get_data()          # (n_ch, n_samples_totais)
    return data[:, -n_need:]       # (n_ch, n_need)


# Pipeline de classificação em cascata (Gating -> Axis -> Direction)
# Os labels são os mesmos definidos no train_subject_model.py:
#   GATING:    0=REST, 1=ACTIVE
#   AXIS:      0=HANDS, 1=FEET
#   DIRECTION: 1=LEFT,  2=RIGHT
def classify_cascade(epoch_data, clf_gate, clf_axis, clf_dir):
    w = epoch_data[np.newaxis, :, :]   # (1, n_ch, n_times)

    # [1] GATING: 0=REST → para aqui
    if clf_gate.predict(w)[0] == 0:
        return 0

    # [2] AXIS: 1=FEET → devolve FEET (3)
    if clf_axis.predict(w)[0] == 1:
        return 3

    # [3] DIRECTION: 1=LEFT, 2=RIGHT
    return int(clf_dir.predict(w)[0])


def send_command(serial_conn, cmd):
    if not serial_conn or not cmd:
        return
    try:
        serial_conn.write((cmd + "\n").encode())
        time.sleep(0.02)
    except Exception:
        pass


# Thread em background que gere a ligação BrainFlow, lê o buffer e dita as fases do ciclo
class RealTimeWorker(QObject):
    status_update = pyqtSignal(str, str)          
    state_update  = pyqtSignal(str, int)          
    prediction_signal = pyqtSignal(int)           
    servo_update  = pyqtSignal(int)               
    log_signal    = pyqtSignal(str)
    finished      = pyqtSignal()

    def __init__(self, session_path):
        super().__init__()
        self.session_path = session_path
        self._stop = False
        self._paused = False
        self.serial_conn = None
        self.board = None

    def stop(self):
        self._stop = True

    def toggle_pause(self, paused):
        self._paused = paused

    def run(self):
        self.log_signal.emit("Loading classification models...")
        try:
            clf_gate = joblib.load(os.path.join(self.session_path, "model_gating.pkl"))
            clf_axis = joblib.load(os.path.join(self.session_path, "model_axis.pkl"))
            clf_dir  = joblib.load(os.path.join(self.session_path, "model_direction.pkl"))
        except Exception as e:
            self.log_signal.emit(f"Model loading error: {e}")
            self.status_update.emit("Model Error", "err")
            self.finished.emit()
            return

        self.log_signal.emit("Connecting to EEG amplifier (BrainFlow)...")
        try:
            params = BrainFlowInputParams()
            params.serial_port = CONFIG["serial_port"]
            self.board = BoardShim(CONFIG["board_id"], params)
            self.board.prepare_session()
            self.board.start_stream()
            sfreq = BoardShim.get_sampling_rate(CONFIG["board_id"])
            eeg_channels = BoardShim.get_eeg_channels(CONFIG["board_id"])
            self.log_signal.emit(f"Amplifier ready! {len(eeg_channels)} channels at {sfreq}Hz.")
        except Exception as e:
            self.log_signal.emit(f"Board error: {e}")
            self.status_update.emit("EEG Error", "err")
            self.finished.emit()
            return

        # Padding extra para o filtro FIR não ter artefactos de borda
        # na janela de classificação. 2 s é seguro para filtros MNE a ~250 Hz.
        FILTER_PAD_SEC = 2.0
        n_need  = int(round(sfreq * WINDOW_SEC))
        n_total = int(round(sfreq * (WINDOW_SEC + FILTER_PAD_SEC)))

        self.status_update.emit("System Active", "ok")
        servo_idx = 0
        self.board.get_board_data()

        while not self._stop:
            if self._paused:
                self.state_update.emit("paused", 0)
                self.status_update.emit("Paused", "warn")
                time.sleep(0.2)
                continue
                
            self.status_update.emit("Running Cycle", "ok")

            # 1. Fase REST
            self.state_update.emit("rest", 0)
            self.log_signal.emit(f"Rest Phase ({REST_DURATION}s)...")
            self.board.get_board_data() 
            
            t_end = time.time() + REST_DURATION
            while time.time() < t_end and not self._stop and not self._paused:
                time.sleep(0.05)
            
            if self._stop or self._paused: continue

            # 2. Fase THINK (Imagética Motora)
            self.state_update.emit("think", 0)
            self.log_signal.emit(f"▸ THINK WINDOW: Control the system now! ({WINDOW_SEC + FILTER_PAD_SEC:.1f}s incl. filter pad)")
            self.board.get_board_data()   # limpa buffer antes de começar
            
            t_end = time.time() + WINDOW_SEC + FILTER_PAD_SEC
            while time.time() < t_end and not self._stop and not self._paused:
                time.sleep(0.05)

            if self._stop or self._paused: continue

            # 3. Classificação
            self.log_signal.emit("Processing buffer...")
            raw_data    = self.board.get_board_data()
            eeg_samples = raw_data[eeg_channels, :]   # (n_ch, n_samples)

            if eeg_samples.shape[1] < n_need:
                self.log_signal.emit("Insufficient samples, skipping trial.")
                continue

            # Passa as últimas n_total amostras (janela + padding);
            # process_buffer_to_epoch descarta o padding depois de filtrar.
            eeg_segment = eeg_samples[:, -min(eeg_samples.shape[1], n_total):]

            try:
                epoch = process_buffer_to_epoch(eeg_segment, sfreq)
                pred  = classify_cascade(epoch, clf_gate, clf_axis, clf_dir)
            except Exception as ex:
                self.log_signal.emit(f"Processing/Prediction error: {ex}")
                continue

            self.prediction_signal.emit(pred)
            self.log_signal.emit(f"➔ Output: {NAMES[pred]} {SYMBOLS[pred]}")

            # 4. Ação do Robot
            self.state_update.emit("move", pred)
            cmd = COMMAND_MAP[pred]
            
            if pred == 3:
                servo_idx = (servo_idx + 1) % N_SERVOS
                self.servo_update.emit(servo_idx)
                send_command(self.serial_conn, "FEET")
                self.log_signal.emit("Active joint switched.")
                time.sleep(MOVE_DURATION) 
            elif cmd is not None:
                t_move_end = time.time() + MOVE_DURATION
                while time.time() < t_move_end and not self._stop:
                    send_command(self.serial_conn, cmd)
                    time.sleep(MOVE_INTERVAL)
            else:
                time.sleep(MOVE_DURATION)

        if self.board:
            try:
                self.board.stop_stream()
                self.board.release_session()
            except Exception:
                pass
                
        self.log_signal.emit("Core runtime worker thread stopped.")
        self.finished.emit()


# Componentes Visuais Customizados: Desenho do braço robótico e indicadores de estado
class ArmDiagram(QWidget):
    def __init__(self):
        super().__init__()
        self.active_idx = 0
        self.setMinimumSize(220, 180)

    def set_active(self, idx):
        self.active_idx = idx
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h   = self.width(), self.height()
        cx, cy = w // 2, h // 2 + 20
        p.fillRect(0, 0, w, h, QColor(BG_PANEL))

        joints = [(cx, cy), (cx + 20, cy - 50), (cx + 55, cy - 95)]

        for i in range(len(joints) - 1):
            col = QColor(COLOR_ACCENT) if i == self.active_idx else QColor(COLOR_BORDER)
            pen = QPen(col, 6 if i == self.active_idx else 3)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(joints[i][0], joints[i][1], joints[i+1][0], joints[i+1][1])

        for i, (jx, jy) in enumerate(joints):
            is_active = (i == self.active_idx)
            r = 9 if is_active else 6
            if is_active:
                p.setPen(QPen(QColor(COLOR_ACCENT + "44"), 4))
                p.drawEllipse(jx - r - 4, jy - r - 4, (r+4)*2, (r+4)*2)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor(COLOR_ACCENT)))
            else:
                p.setPen(QPen(QColor(COLOR_BORDER), 2))
                p.setBrush(QBrush(QColor(BG_INNER)))
            p.drawEllipse(jx - r, jy - r, r*2, r*2)

            p.setPen(QColor(COLOR_ACCENT) if is_active else QColor(COLOR_DIM))
            p.setFont(QFont("Courier New", 8, QFont.Bold if is_active else QFont.Normal))
            p.drawText(jx + r + 5, jy + 4, SERVO_JOINTS[i])
        p.end()


class PredIndicator(QWidget):
    def __init__(self, label, symbol, color, parent=None):
        super().__init__(parent)
        self.label, self.symbol, self.color = label, symbol, color
        self.active = False
        self.setMinimumHeight(50)

    def set_active(self, val):
        self.active = val
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        col  = QColor(self.color)

        p.setBrush(QBrush(QColor(self.color + "44" if self.active else self.color + "0d")))
        p.setPen(QPen(col if self.active else QColor(self.color + "33"), 1.5 if self.active else 1))
        p.drawRoundedRect(1, 1, w-2, h-2, 6, 6)

        p.setPen(col if self.active else QColor(self.color + "55"))
        p.setFont(QFont("Courier New", 11, QFont.Bold if self.active else QFont.Normal))
        p.drawText(0, 0, w, h, Qt.AlignCenter, f"{self.symbol}  {self.label}")
        p.end()


# Classe principal da Interface Gráfica (Janela, Botões e atualização de estados)
class RealTimeWindow(QWidget):
    def __init__(self, session_path):
        super().__init__()
        self.session_path = session_path
        self._serial_conn = None
        self._paused = False

        self.setWindowTitle("BCI Real-Time Robot Controller")
        self.resize(750, 650)
        self.setStyleSheet(f"background-color: {BG_COLOR}; color: {COLOR_TEXT};")

        self._build_ui()
        self._populate_ports()
        
        self._thread = QThread()
        self._worker = RealTimeWorker(self.session_path)
        self._worker.moveToThread(self._thread)
        
        self._thread.started.connect(self._worker.run)
        self._worker.status_update.connect(self._on_status)
        self._worker.state_update.connect(self._on_state)
        self._worker.prediction_signal.connect(self._on_prediction)
        self._worker.servo_update.connect(self._on_servo)
        self._worker.log_signal.connect(self._log)
        self._worker.finished.connect(self._thread.quit)
        
        self._thread.start()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        title = QLabel("REAL-TIME BCI CONTROL SYSTEM")
        title.setFont(QFont("Courier New", 14, QFont.Bold))
        title.setStyleSheet(f"color: {COLOR_ACCENT}; letter-spacing: 3px;")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        root.addWidget(self._hline())

        conn_row = QHBoxLayout()
        conn_row.addWidget(QLabel("Robot Port: "))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(120)
        self.port_combo.setStyleSheet(self._combo_style())
        conn_row.addWidget(self.port_combo)

        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setStyleSheet(self._btn_style(COLOR_ACCENT))
        refresh_btn.clicked.connect(self._populate_ports)
        conn_row.addWidget(refresh_btn)

        self.connect_btn = QPushButton("CONNECT ROBOT")
        self.connect_btn.setStyleSheet(self._btn_style(COLOR_ACCENT))
        self.connect_btn.clicked.connect(self._toggle_connect)
        conn_row.addWidget(self.connect_btn)

        self.conn_status = QLabel("● Disconnected")
        self.conn_status.setStyleSheet(f"color: {COLOR_STOP}; font-weight: bold; font-family: 'Courier New';")
        conn_row.addWidget(self.conn_status)
        
        conn_row.addStretch()
        
        self.status_lbl = QLabel("Loading...")
        self.status_lbl.setFont(QFont("Courier New", 10, QFont.Bold))
        conn_row.addWidget(self.status_lbl)
        root.addLayout(conn_row)

        root.addWidget(self._hline())

        self.phase_box = QFrame()
        self.phase_box.setStyleSheet(f"background-color: {BG_PANEL}; border: 1px solid {COLOR_BORDER}; border-radius: 8px;")
        phase_layout = QVBoxLayout(self.phase_box)
        
        self.phase_title = QLabel("AWAITING START")
        self.phase_title.setFont(QFont("Courier New", 18, QFont.Bold))
        self.phase_title.setAlignment(Qt.AlignCenter)
        self.phase_title.setStyleSheet(f"color: {COLOR_TEXT};")
        phase_layout.addWidget(self.phase_title)
        
        self.phase_desc = QLabel("Initializing acquisition thread...")
        self.phase_desc.setFont(QFont("Courier New", 10))
        self.phase_desc.setAlignment(Qt.AlignCenter)
        self.phase_desc.setStyleSheet(f"color: {COLOR_DIM};")
        phase_layout.addWidget(self.phase_desc)
        root.addWidget(self.phase_box)

        body = QHBoxLayout()
        
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("ACTIVE ROBOT JOINT:"))
        self.active_servo_lbl = QLabel("── BASE ──")
        self.active_servo_lbl.setFont(QFont("Courier New", 12, QFont.Bold))
        self.active_servo_lbl.setStyleSheet(f"color: {COLOR_ACCENT};")
        self.active_servo_lbl.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.active_servo_lbl)
        
        self.arm_diagram = ArmDiagram()
        left_panel.addWidget(self.arm_diagram)
        body.addLayout(left_panel, 3)

        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("CLASSIFIER STAGE OUTPUTS:"))
        self.ind_left  = PredIndicator("LEFT HAND", "←", COLOR_LEFT)
        self.ind_right = PredIndicator("RIGHT HAND", "→", COLOR_RIGHT)
        self.ind_feet  = PredIndicator("FEET (SWITCH JOINT)", "↓", COLOR_FEET)
        self.ind_rest  = PredIndicator("REST", "■", COLOR_REST)
        
        right_panel.addWidget(self.ind_left)
        right_panel.addWidget(self.ind_right)
        right_panel.addWidget(self.ind_feet)
        right_panel.addWidget(self.ind_rest)
        self._indicators = {0: self.ind_rest, 1: self.ind_left, 2: self.ind_right, 3: self.ind_feet}
        body.addLayout(right_panel, 4)
        
        root.addLayout(body)
        root.addWidget(self._hline())

        control_row = QHBoxLayout()
        self.pause_btn = QPushButton("PAUSE SYSTEM")
        self.pause_btn.setFont(QFont("Courier New", 10, QFont.Bold))
        self.pause_btn.setStyleSheet(self._btn_style(COLOR_YELLOW))
        self.pause_btn.clicked.connect(self._toggle_pause)
        control_row.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("STOP & CLOSE")
        self.stop_btn.setFont(QFont("Courier New", 10, QFont.Bold))
        self.stop_btn.setStyleSheet(self._btn_style(COLOR_RED))
        self.stop_btn.clicked.connect(self.close)
        control_row.addWidget(self.stop_btn)
        root.addLayout(control_row)

        root.addWidget(QLabel("SYSTEM CONSOLE LOGS:"))
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumHeight(90)
        self.log_widget.setFont(QFont("Courier New", 8))
        self.log_widget.setStyleSheet(f"background: {BG_PANEL}; color: {COLOR_ACCENT}; border: 1px solid {COLOR_BORDER};")
        root.addWidget(self.log_widget)

    def _populate_ports(self):
        self.port_combo.clear()
        if HAS_SERIAL:
            ports = [p.device for p in serial.tools.list_ports.comports()]
            for p in ports: self.port_combo.addItem(p)
            if not ports: self.port_combo.addItem("No ports detected")
        else:
            self.port_combo.addItem("pyserial missing")

    def _toggle_connect(self):
        if self._serial_conn and self._serial_conn.is_open:
            if self._worker: self._worker.serial_conn = None
            self._serial_conn.close()
            self._serial_conn = None
            self.conn_status.setText("● Disconnected")
            self.conn_status.setStyleSheet(f"color: {COLOR_STOP}; font-weight: bold;")
            self.connect_btn.setText("CONNECT ROBOT")
            self._log("Serial communication closed.")
        else:
            if not HAS_SERIAL: return
            port = self.port_combo.currentText()
            try:
                self._serial_conn = serial.Serial(port, ROBOT_BAUD, timeout=1)
                time.sleep(2) 
                if self._worker: self._worker.serial_conn = self._serial_conn
                self.conn_status.setText("● Connected")
                self.conn_status.setStyleSheet(f"color: {COLOR_ACCENT}; font-weight: bold;")
                self.connect_btn.setText("DISCONNECT")
                self._log(f"Robot connected on port {port}")
            except Exception as e:
                self._log(f"Robot link failure: {e}")

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._worker:
            self._worker.toggle_pause(self._paused)
        if self._paused:
            self.pause_btn.setText("RESUME SYSTEM")
            self.pause_btn.setStyleSheet(self._btn_style(COLOR_GREEN))
        else:
            self.pause_btn.setText("PAUSE SYSTEM")
            self.pause_btn.setStyleSheet(self._btn_style(COLOR_YELLOW))

    def _on_status(self, msg, level):
        colors = {"info": COLOR_DIM, "ok": COLOR_ACCENT, "warn": COLOR_YELLOW, "err": COLOR_RED}
        self.status_lbl.setStyleSheet(f"color: {colors.get(level, COLOR_TEXT)};")
        self.status_lbl.setText(msg.upper())

    def _on_state(self, phase, label_pred):
        if phase == "rest":
            self.phase_box.setStyleSheet(f"background-color: #1c1c28; border: 2px solid {COLOR_REST}; border-radius: 8px;")
            self.phase_title.setText("■ REST / PREPARE")
            self.phase_title.setStyleSheet(f"color: {COLOR_REST};")
            self.phase_desc.setText(f"Relax your mind and stay still for the next {REST_DURATION} seconds.")
            for ind in self._indicators.values(): ind.set_active(False)
        elif phase == "think":
            self.phase_box.setStyleSheet(f"background-color: #11252b; border: 2px solid {COLOR_ACCENT}; border-radius: 8px;")
            self.phase_title.setText("🧠 MOTOR IMAGERY ACTIVE")
            self.phase_title.setStyleSheet(f"color: {COLOR_ACCENT};")
            self.phase_desc.setText(f"THINK WINDOW: Execute your intended motor imagery control strategy ({MI_DURATION}s).")
        elif phase == "move":
            col = CLASS_COLORS.get(label_pred, COLOR_ACCENT)
            self.phase_box.setStyleSheet(f"background-color: {BG_INNER}; border: 2px solid {col}; border-radius: 8px;")
            self.phase_title.setText(f"▶ ROBOT MOVING: {NAMES[label_pred]}")
            self.phase_title.setStyleSheet(f"color: {col};")
            self.phase_desc.setText(f"Streaming continuous commands over serial for {MOVE_DURATION}s.")
        elif phase == "paused":
            self.phase_box.setStyleSheet(f"background-color: #2b2511; border: 2px solid {COLOR_YELLOW}; border-radius: 8px;")
            self.phase_title.setText("⏸ SYSTEM PAUSED")
            self.phase_title.setStyleSheet(f"color: {COLOR_YELLOW};")
            self.phase_desc.setText("Acquisition and execution loops are halted. Click Resume to continue.")

    def _on_prediction(self, pred):
        for cls, ind in self._indicators.items():
            ind.set_active(cls == pred)

    def _on_servo(self, idx):
        self.active_servo_lbl.setText(f"── {SERVO_JOINTS[idx].upper()} ──")
        self.arm_diagram.set_active(idx)

    def _log(self, msg):
        self.log_widget.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _hline(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {COLOR_BORDER};")
        return line

    def _combo_style(self):
        return f"QComboBox {{ background: {BG_PANEL}; color: {COLOR_TEXT}; border: 1px solid {COLOR_BORDER}; padding: 3px; }}"

    def _btn_style(self, color):
        return f"QPushButton {{ background: {color}22; color: {color}; border: 1px solid {color}66; padding: 5px 12px; border-radius: 4px; }} QPushButton:hover {{ background: {color}44; }}"

    def closeEvent(self, event):
        self._log("Shutting down cleanly...")
        if self._worker: self._worker.stop()
        if self._thread:
            self._thread.quit()
            self._thread.wait(2000)
        if self._serial_conn and self._serial_conn.is_open:
            self._serial_conn.close()
        event.accept()


# Ponto de entrada padrão para executar o script e localizar os classificadores gravados
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCI Real-Time Robot Control Window")
    parser.add_argument("session_path", nargs="?", help="Path containing model pkl files from training")
    args = parser.parse_args()

    session_path = args.session_path
    if not session_path:
        data_dir = "data"
        if os.path.isdir(data_dir):
            sessions = sorted([
                os.path.join(data_dir, d) for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d)) and os.path.exists(os.path.join(data_dir, d, "model_gating.pkl"))
            ])
            if sessions:
                session_path = sessions[-1]
                print(f"Auto-detecting latest session: {session_path}")
            else:
                print("Error: No trained model directory found under 'data/'.")
                sys.exit(1)
        else:
            print("Usage: python realtime_robot_control.py <session_path>")
            sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = RealTimeWindow(session_path)
    win.show()
    sys.exit(app.exec_())
# ============================================================
# FILE: robot_control_gui.py
#
# Interface gráfica para controlo manual do braço robótico
# via porta série (insert_brain_connected.ino).
#
# Uso: python robot_control_gui.py
# Dependências: PyQt5, pyserial
# ============================================================

import sys
import serial
import serial.tools.list_ports
import time

from PyQt5.QtCore    import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui     import QFont, QColor, QPainter, QPen, QBrush, QFontDatabase
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QTextEdit,
    QFrame, QGridLayout, QSizePolicy
)

ROBOT_BAUD   = 9600
N_SERVOS     = 3
SERVO_NAMES  = ["Servo 1", "Servo 2", "Servo 3"]
SERVO_JOINTS = ["Base", "Shoulder", "Elbow"]
HOLD_TIMER_INTERVAL = 50   # ms between repeated commands (lower = faster)


BG          = "#0d0d14"
BG2         = "#13131f"
BG3         = "#1a1a2e"
ACCENT      = "#00d4aa"
ACCENT2     = "#4c9be8"
LEFT_COL    = "#4c9be8"
RIGHT_COL   = "#e87a4c"
FEET_COL    = "#a64ce8"
STOP_COL    = "#e84c4c"
TEXT        = "#e0e0f0"
TEXT_DIM    = "#666688"
BORDER      = "#2a2a40"


# ============================================================
# SERIAL WORKER
# ============================================================

class SerialWorker(QThread):
    """Reads Arduino responses in background and emits them as signals."""
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
# SERVO INDICATOR WIDGET
# ============================================================

class ServoIndicator(QWidget):
    """Visual display showing 4 servo joints with active one highlighted."""

    def __init__(self):
        super().__init__()
        self.active = 0
        self.setMinimumSize(240, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_active(self, idx):
        self.active = idx
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h   = self.width(), self.height()
        cx, cy = w // 2, h // 2

        # Background
        p.fillRect(0, 0, w, h, QColor(BG2))

        # Draw arm segments — 3 joints
        joints = [
            (cx,      cy + 60),   # base
            (cx + 20, cy - 10),   # shoulder
            (cx + 60, cy - 65),   # elbow
        ]

        # Draw segments between joints
        for i in range(len(joints) - 1):
            is_active_seg = (i == self.active or i + 1 == self.active)
            color = QColor(ACCENT) if i == self.active else QColor(BORDER)
            pen = QPen(color, 6 if is_active_seg else 3)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(joints[i][0], joints[i][1],
                       joints[i+1][0], joints[i+1][1])

        # Draw joint circles
        for i, (jx, jy) in enumerate(joints):
            is_active = (i == self.active)
            r = 10 if is_active else 6

            if is_active:
                # Glow ring
                p.setPen(QPen(QColor(ACCENT + "55"), 3))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(jx - r - 4, jy - r - 4, (r + 4) * 2, (r + 4) * 2)

                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor(ACCENT)))
            else:
                p.setPen(QPen(QColor(BORDER), 2))
                p.setBrush(QBrush(QColor(BG3)))

            p.drawEllipse(jx - r, jy - r, r * 2, r * 2)

            # Label
            lbl_color = QColor(ACCENT) if is_active else QColor(TEXT_DIM)
            p.setPen(lbl_color)
            p.setFont(QFont("Courier New", 8, QFont.Bold if is_active else QFont.Normal))
            p.drawText(jx + r + 4, jy + 4, SERVO_JOINTS[i])

        p.end()


# ============================================================
# MAIN WINDOW
# ============================================================

class RobotControlWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Insert-Brain — Robot Control")
        self.resize(680, 560)
        self.setStyleSheet(f"background-color: {BG}; color: {TEXT};")

        self.serial_port  = None
        self.serial_worker = None
        self.active_servo = 0     # mirrors Arduino state

        self._hold_timer = QTimer()
        self._hold_timer.setInterval(HOLD_TIMER_INTERVAL)   # ms between repeated commands — lower = faster
        self._hold_timer.timeout.connect(self._on_hold_tick)
        self._hold_cmd = None

        self._build_ui()
        self._populate_ports()

    # ── Build UI ──────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        # ── Header
        title = QLabel("ROBOT ARM CONTROL")
        title.setFont(QFont("Courier New", 16, QFont.Bold))
        title.setStyleSheet(f"color: {ACCENT}; letter-spacing: 4px;")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel("manual override interface")
        subtitle.setFont(QFont("Courier New", 9))
        subtitle.setStyleSheet(f"color: {TEXT_DIM}; letter-spacing: 2px;")
        subtitle.setAlignment(Qt.AlignCenter)
        root.addWidget(subtitle)

        root.addWidget(self._hline())

        # ── Connection row
        conn_row = QHBoxLayout()

        self.port_combo = QComboBox()
        self.port_combo.setStyleSheet(self._combo_style())
        self.port_combo.setMinimumWidth(140)

        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(32)
        refresh_btn.setStyleSheet(self._small_btn_style(ACCENT))
        refresh_btn.clicked.connect(self._populate_ports)
        refresh_btn.setToolTip("Refresh ports")

        self.connect_btn = QPushButton("CONNECT")
        self.connect_btn.setFixedWidth(110)
        self.connect_btn.setStyleSheet(self._action_btn_style(ACCENT))
        self.connect_btn.clicked.connect(self._toggle_connect)

        self.conn_status = QLabel("● Disconnected")
        self.conn_status.setStyleSheet(f"color: {STOP_COL}; font-family: 'Courier New'; font-size: 11px;")

        conn_row.addWidget(QLabel("Port:"))
        conn_row.addWidget(self.port_combo)
        conn_row.addWidget(refresh_btn)
        conn_row.addSpacing(8)
        conn_row.addWidget(self.connect_btn)
        conn_row.addSpacing(12)
        conn_row.addWidget(self.conn_status)
        conn_row.addStretch()
        root.addLayout(conn_row)

        root.addWidget(self._hline())

        # ── Main body: arm diagram + controls
        body = QHBoxLayout()
        body.setSpacing(16)

        # Left: arm diagram + active servo label
        left_panel = QVBoxLayout()

        active_lbl_title = QLabel("ACTIVE SERVO")
        active_lbl_title.setFont(QFont("Courier New", 9))
        active_lbl_title.setStyleSheet(f"color: {TEXT_DIM}; letter-spacing: 2px;")
        active_lbl_title.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(active_lbl_title)

        self.active_lbl = QLabel(f"── {SERVO_JOINTS[0].upper()} ──")
        self.active_lbl.setFont(QFont("Courier New", 14, QFont.Bold))
        self.active_lbl.setStyleSheet(f"color: {ACCENT}; letter-spacing: 3px;")
        self.active_lbl.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.active_lbl)

        self.servo_indicator = ServoIndicator()
        left_panel.addWidget(self.servo_indicator)

        # Servo number pills
        pill_row = QHBoxLayout()
        pill_row.setSpacing(6)
        self.servo_pills = []
        for i in range(N_SERVOS):
            pill = QLabel(f"S{i+1}")
            pill.setAlignment(Qt.AlignCenter)
            pill.setFixedSize(36, 26)
            pill.setFont(QFont("Courier New", 9, QFont.Bold))
            self.servo_pills.append(pill)
            pill_row.addWidget(pill)
        self._refresh_pills()
        left_panel.addLayout(pill_row)
        left_panel.addStretch()

        body.addLayout(left_panel, 2)

        # Vertical divider
        div = QFrame()
        div.setFrameShape(QFrame.VLine)
        div.setStyleSheet(f"color: {BORDER};")
        body.addWidget(div)

        # Right: control buttons
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        ctrl_title = QLabel("COMMANDS")
        ctrl_title.setFont(QFont("Courier New", 9))
        ctrl_title.setStyleSheet(f"color: {TEXT_DIM}; letter-spacing: 2px;")
        right_panel.addWidget(ctrl_title)

        # LEFT button
        self.left_btn = QPushButton("← LEFT HAND")
        self.left_btn.setFixedHeight(56)
        self.left_btn.setFont(QFont("Courier New", 12, QFont.Bold))
        self.left_btn.setStyleSheet(self._cmd_btn_style(LEFT_COL))
        self.left_btn.pressed.connect(lambda: self._start_hold("LEFT"))
        self.left_btn.released.connect(self._stop_hold)
        right_panel.addWidget(self.left_btn)

        hint_left = QLabel("active servo  ←  backward")
        hint_left.setStyleSheet(f"color: {TEXT_DIM}; font-family: 'Courier New'; font-size: 9px;")
        hint_left.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(hint_left)

        right_panel.addSpacing(4)

        # RIGHT button
        self.right_btn = QPushButton("RIGHT HAND →")
        self.right_btn.setFixedHeight(56)
        self.right_btn.setFont(QFont("Courier New", 12, QFont.Bold))
        self.right_btn.setStyleSheet(self._cmd_btn_style(RIGHT_COL))
        self.right_btn.pressed.connect(lambda: self._start_hold("RIGHT"))
        self.right_btn.released.connect(self._stop_hold)
        right_panel.addWidget(self.right_btn)

        hint_right = QLabel("active servo  →  forward")
        hint_right.setStyleSheet(f"color: {TEXT_DIM}; font-family: 'Courier New'; font-size: 9px;")
        hint_right.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(hint_right)

        right_panel.addSpacing(4)

        # FEET button
        self.feet_btn = QPushButton("↕  FEET  —  SWITCH SERVO")
        self.feet_btn.setFixedHeight(56)
        self.feet_btn.setFont(QFont("Courier New", 12, QFont.Bold))
        self.feet_btn.setStyleSheet(self._cmd_btn_style(FEET_COL))
        self.feet_btn.clicked.connect(lambda: self._send_cmd("FEET"))
        right_panel.addWidget(self.feet_btn)

        hint_feet = QLabel(f"cycles:  Base → Shoulder → Elbow")
        hint_feet.setStyleSheet(f"color: {TEXT_DIM}; font-family: 'Courier New'; font-size: 9px;")
        hint_feet.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(hint_feet)

        right_panel.addSpacing(8)
        right_panel.addWidget(self._hline())
        right_panel.addSpacing(4)

        # STOP button
        self.stop_btn = QPushButton("■  STOP — NEUTRAL POSITION")
        self.stop_btn.setFixedHeight(40)
        self.stop_btn.setFont(QFont("Courier New", 10, QFont.Bold))
        self.stop_btn.setStyleSheet(self._cmd_btn_style(STOP_COL, outline=True))
        self.stop_btn.clicked.connect(lambda: self._send_cmd("STOP"))
        right_panel.addWidget(self.stop_btn)

        right_panel.addStretch()
        body.addLayout(right_panel, 3)
        root.addLayout(body)

        root.addWidget(self._hline())

        # ── Serial log
        log_lbl = QLabel("SERIAL LOG")
        log_lbl.setFont(QFont("Courier New", 8))
        log_lbl.setStyleSheet(f"color: {TEXT_DIM}; letter-spacing: 2px;")
        root.addWidget(log_lbl)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        self.log.setFont(QFont("Courier New", 9))
        self.log.setStyleSheet(
            f"background: {BG2}; color: {ACCENT}; "
            f"border: 1px solid {BORDER}; border-radius: 4px; padding: 4px;"
        )
        root.addWidget(self.log)

        self._set_controls_enabled(False)

    # ── Serial connection ──────────────────────────────────────

    def _populate_ports(self):
        self.port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        for p in ports:
            self.port_combo.addItem(p)
        if not ports:
            self.port_combo.addItem("No ports found")

    def _toggle_connect(self):
        if self.serial_port and self.serial_port.is_open:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self.port_combo.currentText()
        try:
            self.serial_port = serial.Serial(port, ROBOT_BAUD, timeout=1)
            time.sleep(2)   # Arduino reset
            self.serial_worker = SerialWorker(self.serial_port)
            self.serial_worker.line_received.connect(self._on_serial_line)
            self.serial_worker.start()
            self.conn_status.setText("● Connected")
            self.conn_status.setStyleSheet(f"color: {ACCENT}; font-family: 'Courier New'; font-size: 11px;")
            self.connect_btn.setText("DISCONNECT")
            self._set_controls_enabled(True)
            self._log(f"Connected to {port} @ {ROBOT_BAUD} baud")
        except serial.SerialException as e:
            self._log(f"ERROR: {e}")

    def _disconnect(self):
        if self.serial_worker:
            self.serial_worker.stop()
            self.serial_worker.wait()
            self.serial_worker = None
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.conn_status.setText("● Disconnected")
        self.conn_status.setStyleSheet(f"color: {STOP_COL}; font-family: 'Courier New'; font-size: 11px;")
        self.connect_btn.setText("CONNECT")
        self._set_controls_enabled(False)
        self._log("Disconnected.")

    # ── Commands ──────────────────────────────────────────────

    def _start_hold(self, cmd):
        self._hold_cmd = cmd
        self._send_cmd(cmd)        # fire immediately on press
        self._hold_timer.start()   # then repeat every interval

    def _stop_hold(self):
        self._hold_timer.stop()
        self._hold_cmd = None

    def _on_hold_tick(self):
        if self._hold_cmd:
            self._send_cmd(self._hold_cmd)
        
    def _send_cmd(self, cmd):
        if not self.serial_port or not self.serial_port.is_open:
            self._log("Not connected.")
            return
        try:
            self.serial_port.write((cmd + "\n").encode())
            self._log(f"→ {cmd}")

            if cmd == "FEET":
                self.active_servo = (self.active_servo + 1) % N_SERVOS
                self._refresh_active()
            elif cmd == "STOP":
                self.active_servo = 0
                self._refresh_active()
        except serial.SerialException as e:
            self._log(f"ERROR: {e}")

    def _on_serial_line(self, line):
        self._log(f"← {line}")

    # ── UI helpers ────────────────────────────────────────────

    def _refresh_active(self):
        joint = SERVO_JOINTS[self.active_servo]
        self.active_lbl.setText(f"── {joint.upper()} ──")
        self.servo_indicator.set_active(self.active_servo)
        self._refresh_pills()

    def _refresh_pills(self):
        for i, pill in enumerate(self.servo_pills):
            if i == self.active_servo:
                pill.setStyleSheet(
                    f"background: {ACCENT}22; color: {ACCENT}; "
                    f"border: 1px solid {ACCENT}; border-radius: 4px;"
                )
            else:
                pill.setStyleSheet(
                    f"background: {BG2}; color: {TEXT_DIM}; "
                    f"border: 1px solid {BORDER}; border-radius: 4px;"
                )

    def _set_controls_enabled(self, enabled):
        for btn in (self.left_btn, self.right_btn, self.feet_btn, self.stop_btn):
            btn.setEnabled(enabled)
            opacity = "1.0" if enabled else "0.35"
            btn.setStyleSheet(btn.styleSheet())   # force repaint

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}]  {msg}")

    def _hline(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {BORDER};")
        return line

    # ── Styles ────────────────────────────────────────────────

    def _cmd_btn_style(self, color, outline=False):
        if outline:
            return (
                f"QPushButton {{"
                f"  background: transparent;"
                f"  color: {color};"
                f"  border: 1px solid {color}88;"
                f"  border-radius: 6px;"
                f"  letter-spacing: 2px;"
                f"}}"
                f"QPushButton:hover {{"
                f"  background: {color}22;"
                f"  border: 1px solid {color};"
                f"}}"
                f"QPushButton:pressed {{"
                f"  background: {color}44;"
                f"}}"
                f"QPushButton:disabled {{"
                f"  color: {color}44;"
                f"  border-color: {color}22;"
                f"}}"
            )
        return (
            f"QPushButton {{"
            f"  background: {color}18;"
            f"  color: {color};"
            f"  border: 1px solid {color}66;"
            f"  border-radius: 6px;"
            f"  letter-spacing: 2px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background: {color}33;"
            f"  border: 1px solid {color}cc;"
            f"}}"
            f"QPushButton:pressed {{"
            f"  background: {color}55;"
            f"}}"
            f"QPushButton:disabled {{"
            f"  background: {color}08;"
            f"  color: {color}33;"
            f"  border-color: {color}18;"
            f"}}"
        )

    def _action_btn_style(self, color):
        return (
            f"QPushButton {{"
            f"  background: {color}22; color: {color};"
            f"  border: 1px solid {color}66; border-radius: 4px;"
            f"  padding: 4px 10px; font-family: 'Courier New'; font-size: 10px;"
            f"  letter-spacing: 1px;"
            f"}}"
            f"QPushButton:hover {{ background: {color}44; border-color: {color}; }}"
        )

    def _small_btn_style(self, color):
        return (
            f"QPushButton {{"
            f"  background: {color}22; color: {color};"
            f"  border: 1px solid {color}55; border-radius: 4px;"
            f"  font-size: 14px;"
            f"}}"
            f"QPushButton:hover {{ background: {color}44; }}"
        )

    def _combo_style(self):
        return (
            f"QComboBox {{"
            f"  background: {BG2}; color: {TEXT};"
            f"  border: 1px solid {BORDER}; border-radius: 4px;"
            f"  padding: 4px 8px; font-family: 'Courier New'; font-size: 10px;"
            f"}}"
            f"QComboBox::drop-down {{ border: none; }}"
            f"QComboBox QAbstractItemView {{"
            f"  background: {BG2}; color: {TEXT}; border: 1px solid {BORDER};"
            f"}}"
        )

    # ── Cleanup ───────────────────────────────────────────────

    def closeEvent(self, event):
        self._disconnect()
        event.accept()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = RobotControlWindow()
    win.show()
    sys.exit(app.exec_())
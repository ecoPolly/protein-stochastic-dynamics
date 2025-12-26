import numpy as np
import sys
import json
import os
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

import matplotlib.pyplot as plt
plt.style.use('dark_background')
from IPython.display import display, HTML
display(HTML("<style>.container{width:95% !important;}</style>"))

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt5.uic import loadUiType
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import savgol_filter


UI_FILE = "UI_pulse_check.ui"
Ui_MainWindow, QtBaseClass = loadUiType(UI_FILE)
file_path = ""


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('RAMP ANALYZER')
        
        self._configure_matplotlib()
        self._apply_dark_theme()
        self._init_state()
        self._setup_plots()
        self._setup_controls()
        self._connect_signals()

    def _configure_matplotlib(self):
        plt.rcParams.update({
            "figure.facecolor": "#2b2b2b",
            "axes.facecolor": "#2f2f2f",
            "savefig.facecolor": "#2b2b2b",
            "axes.edgecolor": "#cfcfcf",
            "axes.labelcolor": "#e6e6e6",
            "xtick.color": "#e6e6e6",
            "ytick.color": "#e6e6e6",
            "grid.color": "#565656",
            "text.color": "#e6e6e6",
            "xtick.labelsize": 6,
            "ytick.labelsize": 6
        })

    def _apply_dark_theme(self):
        dark_style = """
        QWidget { background: #2b2b2b; color: #e6e6e6; }
        QLabel { color: #e6e6e6; }
        QPushButton {
            background: #3a3a3a; color: #e6e6e6; border: 1px solid #4a4a4a;
            padding: 6px 10px; border-radius: 6px;
        }
        QPushButton:hover { background: #444; }
        QPushButton:pressed { background: #393939; }
        QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {
            background: #242424; color: #e6e6e6; border: 1px solid #4a4a4a;
            border-radius: 4px;
        }
        """
        app.setStyleSheet(dark_style)

    def _init_state(self):
        self.current_pulse_idx = 0
        self.total_pulses = 0
        self.valid_pulses = []
        self.rejected_pulses = []
        self.data_cache = None

    def _setup_plots(self):
        # Extension plot
        canvas1 = FigureCanvas(Figure(constrained_layout=True, edgecolor='black', frameon=True))
        self.Plotting_box_3.addWidget(canvas1)
        self.grafico = canvas1.figure.subplots()
        self.grafico.set_ylabel('Extension (nm)', size=6)
        self.grafico.set_xlabel('Time (s)', size=6)
        self.line_raw, = self.grafico.plot([], [], lw=1, alpha=0.7)
        self.line_smth, = self.grafico.plot([], [], lw=1)

        # Force plot
        canvas2 = FigureCanvas(Figure(constrained_layout=True, edgecolor='black', frameon=True))
        self.Plotting_box_4.addWidget(canvas2)
        self.grafico2 = canvas2.figure.subplots()
        self.grafico2.set_xlabel('Time (s)', size=6)
        self.grafico2.set_ylabel('Force (pN)', size=6)
        self.line_force, = self.grafico2.plot([], [], lw=1)

    def _setup_controls(self):
        self.btn_accept = self._create_button("✓ Accept Pulse (Space)", "Space")
        self.btn_reject = self._create_button("✗ Reject Pulse (R)", "R")
        self.btn_previous = self._create_button("← Previous", "Left")
        self.btn_next = self._create_button("Next →", "Right")
        self.btn_save = QPushButton("Save Filtered Data")

        self.lbl_progress = QLabel("Pulse 1 / 0")
        self.lbl_accepted = QLabel("Accepted: 0")
        self.lbl_rejected = QLabel("Rejected: 0")

        layout = QVBoxLayout()
        for widget in [self.btn_accept, self.btn_reject, self.btn_previous, 
                      self.btn_next, self.btn_save, self.lbl_progress, 
                      self.lbl_accepted, self.lbl_rejected]:
            layout.addWidget(widget)

        self.controls_box.setLayout(layout)

    def _create_button(self, text, shortcut):
        btn = QPushButton(text)
        btn.setShortcut(shortcut)
        return btn

    def _connect_signals(self):
        self.Open_JSON.clicked.connect(self.file_open)
        self.Plot_button.clicked.connect(self.plot_pulse)
        self.btn_accept.clicked.connect(self.accept_pulse)
        self.btn_reject.clicked.connect(self.reject_pulse)
        self.btn_previous.clicked.connect(self.previous_pulse)
        self.btn_next.clicked.connect(self.next_pulse)
        self.btn_save.clicked.connect(self.save_filtered_data)

    def file_open(self):
        global file_path
        name = QtWidgets.QFileDialog.getOpenFileName(self, "Open JSON File")
        file_path = name[0]
        return file_path

    def load_json(self):
        if self.data_cache is not None:
            return self.data_cache

        with open(file_path, 'r') as f:
            self.data = json.load(f)

        xs, forces, times = [], [], []
        for i in range(len(self.data)):
            pulse_data = self.data[f"Pulse_Number_{i}"]
            z = np.array(pulse_data["z"])
            fz = np.array(pulse_data["force"])
            t = np.array(pulse_data["time"]) - pulse_data["time"][0]
            
            xs.append(z)
            forces.append(fz)
            times.append(t)

        self.total_pulses = len(xs)
        self.Output.setText(f"Number of pulses: {self.total_pulses}")

        xs_smth = [savgol_filter(x, 51, 4) for x in xs]
        self.data_cache = (xs, forces, times, xs_smth)
        return self.data_cache

    def update_plots(self, idx):
        xs, forces, times, xs_smth = self.load_json()
        
        self.line_raw.set_data(times[idx], xs[idx])
        self.line_smth.set_data(times[idx], xs_smth[idx])
        self.line_force.set_data(times[idx], forces[idx])

        for graph in [self.grafico, self.grafico2]:
            graph.relim()
            graph.autoscale_view()
            graph.figure.canvas.draw_idle()

    def plot_pulse(self):
        self.current_pulse_idx = self.Pulse_num.value()
        self.update_plots(self.current_pulse_idx)

    def accept_pulse(self):
        self._classify_pulse(accept=True)

    def reject_pulse(self):
        self._classify_pulse(accept=False)

    def _classify_pulse(self, accept):
        idx = self.current_pulse_idx
        
        if accept:
            self.rejected_pulses = [p for p in self.rejected_pulses if p != idx]
            if idx not in self.valid_pulses:
                self.valid_pulses.append(idx)
        else:
            self.valid_pulses = [p for p in self.valid_pulses if p != idx]
            if idx not in self.rejected_pulses:
                self.rejected_pulses.append(idx)

        self._update_ui()
        self.update_plots(self.current_pulse_idx)
        self._auto_next()

    def previous_pulse(self):
        if self.current_pulse_idx > 0:
            self.current_pulse_idx -= 1
            self._update_ui()
            self.update_plots(self.current_pulse_idx)

    def next_pulse(self):
        if self.current_pulse_idx < self.total_pulses - 1:
            self.current_pulse_idx += 1
            self._update_ui()
            self.update_plots(self.current_pulse_idx)

    def _auto_next(self):
        classified = set(self.valid_pulses + self.rejected_pulses)
        
        for i in range(self.current_pulse_idx + 1, self.total_pulses):
            if i not in classified:
                self.current_pulse_idx = i
                self._update_ui()
                self.update_plots(self.current_pulse_idx)
                return
        
        QMessageBox.information(self, "Completed",
                              f"All pulses classified!\n"
                              f"Accepted: {len(self.valid_pulses)}\n"
                              f"Rejected: {len(self.rejected_pulses)}")

    def _update_ui(self):
        self.lbl_progress.setText(f"Pulse {self.current_pulse_idx + 1} / {self.total_pulses}")
        self.lbl_accepted.setText(f"Accepted: {len(self.valid_pulses)}")
        self.lbl_rejected.setText(f"Rejected: {len(self.rejected_pulses)}")

    def save_filtered_data(self):
        filtered = {
            f"Pulse_Number_{new_idx}": self.data[f"Pulse_Number_{old_idx}"]
            for new_idx, old_idx in enumerate(sorted(self.valid_pulses))
        }
        
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Filtered Data", "", "JSON Files (*.json)")
        
        if save_path:
            with open(save_path, "w") as f:
                json.dump(filtered, f, indent=2)
            QMessageBox.information(self, "Saved", f"Filtered data saved to {save_path}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
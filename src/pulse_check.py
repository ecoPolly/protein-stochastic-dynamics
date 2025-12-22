import numpy as np
import sys
import scipy
import json
import os
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

import matplotlib.pyplot as plt
import csv
plt.style.use('dark_background')
from IPython.display import display, HTML
display(HTML("<style>.container{width:95% !important;}</style>"))

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, QSlider, QPushButton, QLCDNumber, QVBoxLayout, QMessageBox
from PyQt5.QtCore import QSize
from PyQt5.uic import loadUiType
from PyQt5.QtGui import QIcon

from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from scipy.signal import savgol_filter


qtCreatorFile = "UI_pulse_check.ui"  # File creato con Qt Designer
Ui_MainWindow, QtBaseClass = loadUiType(qtCreatorFile)
Unfolding_forces = []
global file_path
file_path = ""


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('RAMP ANALYZER')

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
        dark = """
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
        app.setStyleSheet(dark)

        self.current_pulse_idx = 0
        self.total_pulses = 0
        self.valid_pulses = []
        self.rejected_pulses = []
        self.data_cache = None

        self.setup_manual_controls()
        self.Open_JSON.clicked.connect(self.File_open)
        self.Plot_button.clicked.connect(self.Plotter)

        grafico=FigureCanvas(Figure(constrained_layout=True, edgecolor='black', frameon=True))
        self.Plotting_box_3.addWidget(grafico)
        self.grafico=grafico.figure.subplots()
        self.grafico.set_ylabel('Extention(nm)', size=6)
        self.grafico.set_xlabel('time (s)', size=6)
        
        x=np.linspace(0,0.03, 100)
        self.grafico.plot(x, 0*x)
        
        grafico2=FigureCanvas(Figure(constrained_layout=True, edgecolor='black', frameon=True))
        self.Plotting_box_4.addWidget(grafico2)
        self.grafico2=grafico2.figure.subplots()
        self.grafico2.set_xlabel('time (s)', size=6)
        self.grafico2.set_ylabel('Force (pN)', size=6)

        # Linee grafico 1 (estensione)
        (self.line_raw,) = self.grafico.plot([], [], lw=1, alpha=0.7)
        (self.line_smth,) = self.grafico.plot([], [], lw=1)

        # Linea grafico 2 (forza)
        (self.line_force,) = self.grafico2.plot([], [], lw=1)



    # -----------------------------
    #  AGGIUNTA PULSANTI MANUALI
    # -----------------------------
    def setup_manual_controls(self):
        """Crea i pulsanti e li inserisce nel box previsto in Qt Designer"""

        # Pulsanti
        self.btn_accept = QPushButton("✓ Accept Pulse (Tab)")
        self.btn_reject = QPushButton("✗ Reject Pulse (r)")
        self.btn_previous = QPushButton("← Previous")
        self.btn_next = QPushButton("Next →")
        self.btn_save_filtered = QPushButton("💾 Save Filtered Data")

        # Labels informativi
        self.lbl_progress = QLabel("Pulse 1 / 0")
        self.lbl_accepted = QLabel("Accepted: 0")
        self.lbl_rejected = QLabel("Rejected: 0")

        # Connetti i pulsanti
        self.btn_accept.clicked.connect(self.accept_pulse)
        self.btn_reject.clicked.connect(self.reject_pulse)
        self.btn_previous.clicked.connect(self.previous_pulse)
        self.btn_next.clicked.connect(self.next_pulse)
        self.btn_save_filtered.clicked.connect(self.save_filtered_data)

        # Scorciatoie
        self.btn_accept.setShortcut("Space")
        self.btn_reject.setShortcut("R")
        self.btn_next.setShortcut("Right")
        self.btn_previous.setShortcut("Left")

        # Layout verticale (inserito in un QVBoxLayout nella UI Designer)
        layout = QVBoxLayout()
        layout.addWidget(self.btn_accept)
        layout.addWidget(self.btn_reject)
        layout.addWidget(self.btn_previous)
        layout.addWidget(self.btn_next)
        layout.addWidget(self.btn_save_filtered)
        layout.addWidget(self.lbl_progress)
        layout.addWidget(self.lbl_accepted)
        layout.addWidget(self.lbl_rejected)

        # Inserisci nel box placeholder definito nel .ui
        self.controls_box.setLayout(layout)  # <-- 'controls_box' deve essere un QWidget nel .ui

    def File_open(self):
        name = QtWidgets.QFileDialog.getOpenFileName(self, "Open JSON File")
        global file_path
        file_path = name[0]
        return file_path

    def Load_JSON(self):
        if self.data_cache is not None:
            return self.data_cache  # usa la cache se già caricata

        with open(file_path, 'r') as f:
            self.data = json.load(f)

        xs, forces, times = [], [], []
        for i in range(len(self.data)):
            z = np.array(self.data[f"Pulse_Number_{i}"]["z"])
            fz = np.array(self.data[f"Pulse_Number_{i}"]["force"])
            t = np.array(self.data[f"Pulse_Number_{i}"]["time"])
            t = t - t[0]
            xs.append(z)
            forces.append(fz)
            times.append(t)

        self.total_pulses = len(xs)
        self.Output.setText(f"Number of pulses: {self.total_pulses}")

        # Precalcola i smooth e salva tutto in cache
        xs_smth = [savgol_filter(x, 51, 4) for x in xs]
        self.data_cache = (xs, forces, times, xs_smth)

        return self.data_cache
    
    def update_plots(self, i, xs, forces, times, xs_smth):
        # Aggiorna linee senza ridisegnare da zero
        self.line_raw.set_data(times[i], xs[i])
        self.line_smth.set_data(times[i], xs_smth[i])
        self.line_force.set_data(times[i], forces[i])

        self.grafico.relim()
        self.grafico.autoscale_view()
        self.grafico2.relim()
        self.grafico2.autoscale_view()

        # Disegna in modo "lazy"
        self.grafico.figure.canvas.draw_idle()
        self.grafico2.figure.canvas.draw_idle()


    def Plotter(self):
        xs, forces, times, xs_smth = self.Load_JSON()
        i = self.Pulse_num.value()
        self.current_pulse_idx = i
        self.update_plots(i, xs, forces, times, xs_smth)


    def accept_pulse(self):
        if self.current_pulse_idx in self.rejected_pulses:
            self.rejected_pulses.remove(self.current_pulse_idx)
        if self.current_pulse_idx not in self.valid_pulses:
            self.valid_pulses.append(self.current_pulse_idx)

        self.update_ui()
        self.show_current_pulse()
        self.auto_next()

    def reject_pulse(self):
        if self.current_pulse_idx in self.valid_pulses:
            self.valid_pulses.remove(self.current_pulse_idx)
        if self.current_pulse_idx not in self.rejected_pulses:
            self.rejected_pulses.append(self.current_pulse_idx)

        self.update_ui()
        self.show_current_pulse()
        self.auto_next()

    def previous_pulse(self):
        if self.current_pulse_idx > 0:
            self.current_pulse_idx -= 1
            self.update_ui()
            self.show_current_pulse()

    def next_pulse(self):
        if self.current_pulse_idx < self.total_pulses - 1:
            self.current_pulse_idx += 1
            self.update_ui()
            self.show_current_pulse()

    def auto_next(self):
        for i in range(self.current_pulse_idx + 1, self.total_pulses):
            if i not in self.valid_pulses and i not in self.rejected_pulses:
                self.current_pulse_idx = i
                self.update_ui()
                self.show_current_pulse()
                return
        QMessageBox.information(self, "Completed",
                                f"All pulses classified!\n"
                                f"Accepted: {len(self.valid_pulses)}\n"
                                f"Rejected: {len(self.rejected_pulses)}")

    def update_ui(self):
        self.lbl_progress.setText(f"Pulse {self.current_pulse_idx + 1} / {self.total_pulses}")
        self.lbl_accepted.setText(f"Accepted: {len(self.valid_pulses)}")
        self.lbl_rejected.setText(f"Rejected: {len(self.rejected_pulses)}")

    def show_current_pulse(self):
        xs, forces, times, xs_smth = self.Load_JSON()
        i = self.current_pulse_idx
        self.update_plots(i, xs, forces, times, xs_smth)

    def save_filtered_data(self):
        xs, forces, times, xs_smth = self.Load_JSON()
        filtered = {
            f"Pulse_Number_{new_idx}": self.data[f"Pulse_Number_{old_idx}"]
            for new_idx, old_idx in enumerate(sorted(self.valid_pulses))}
        
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Filtered Data", "", "JSON Files (*.json)")
        if save_path:
            with open(save_path, "w") as f:
                json.dump(filtered, f, indent=2)
            QMessageBox.information(self, "Saved", f"Filtered data saved to {save_path}")



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())

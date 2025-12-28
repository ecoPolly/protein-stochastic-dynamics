"""
RAMP ANALYZER - Protein Unfolding Analysis Tool

Analyzes force spectroscopy data from JSON files containing pulse measurements.
Detects unfolding events, calculates transition points, and fits Bell-Evans model.

Input: JSON file with pulse data (z, force, time, current)
Output: Unfolding force distributions, transition analysis, WLC fits

Features:
- Dual Event Checkbox: Toggle between single-process (unfolding only) and 
  dual-process (folding + unfolding) ramp analysis
- Transition Detection: Filters ramps based on minimum extension threshold 
  (default: 20 nm for Talin R3 unfolding length)
"""

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
from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUiType
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


UI_FILE = "UI_ramp_analysis.ui"
Ui_MainWindow, QtBaseClass = loadUiType(UI_FILE)

# Constants
KBT = 4.114  # pN·nm
CURRENT_TO_FORCE_A = 1.504e-5
CURRENT_TO_FORCE_B = -0.0133


def bell_model(F, x_beta, k0, r):
    """Bell-Evans model for force distribution"""
    F = np.asarray(F)
    exponent = np.clip((F * x_beta) / KBT, -700, 700)
    exp_term = np.exp(exponent)
    suppression = np.exp(-((k0 * KBT) / (r * x_beta)) * (exp_term - 1))
    return (k0 / r) * exp_term * suppression


def fit_bell_evans_model(forces, r):
    """Fit Bell-Evans model to force distribution"""
    hist, bin_edges = np.histogram(forces, bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    popt, _ = curve_fit(lambda F, x, k0: bell_model(F, x, k0, r), bin_centers, hist)
    x_beta, k0 = popt
    x_fit = np.linspace(min(forces), max(forces), 200)
    y_fit = bell_model(x_fit, x_beta, k0, r)
    return x_beta, k0, x_fit, y_fit, bin_centers, hist


def detect_jump(xs_smth, threshold, drift=0.02):
    """CUSUM algorithm for jump detection"""
    s_pos = np.zeros(len(xs_smth))
    s_neg = np.zeros(len(xs_smth))
    for i in range(1, len(xs_smth)):
        diff = xs_smth[i] - xs_smth[i - 1]
        s_pos[i] = max(0, s_pos[i - 1] + diff - drift)
        s_neg[i] = min(0, s_neg[i - 1] + diff + drift)
        if s_pos[i] > threshold or abs(s_neg[i]) > threshold:
            return i
    return None


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('RAMP ANALYZER')
        
        self._configure_matplotlib()
        self._apply_dark_theme()
        self._init_state()
        self._setup_plots()
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
        """Initialize data cache and state variables"""
        self.file_path = ""
        self.data = None
        self.xs_cache = None
        self.forces_cache = None
        self.times_cache = None
        self.xs_smth_cache = None
        self.unfolding_forces = []
        self.Output.setText("Ready")

    def _setup_plots(self):
        """Setup all plotting canvases"""
        # Force vs Extension
        canvas1 = FigureCanvas(Figure(figsize=(8, 6), constrained_layout=True, 
                                     edgecolor='black', frameon=True))
        self.Plotting_box.addWidget(canvas1)
        self.grafico = canvas1.figure.subplots()
        self.grafico.set_ylabel('Force (pN)', size=6)
        self.grafico.set_xlabel('Extension (nm)', size=6)

        # Unfolding Force Histogram
        canvas3 = FigureCanvas(Figure(figsize=(8, 6), constrained_layout=True,
                                     edgecolor='black', frameon=True))
        self.Plotting_box_3.addWidget(canvas3)
        self.grafico3 = canvas3.figure.subplots()
        self.grafico3.set_xlabel('Unfolding Force (pN)', size=6)
        self.grafico3.set_ylabel('Counts', size=6)

        # Jump Height vs Unfolding Force correlation
        canvas4 = FigureCanvas(Figure(figsize=(8, 6), constrained_layout=True,
                                     edgecolor='black', frameon=True))
        self.Plotting_box_4.addWidget(canvas4)
        self.grafico4 = canvas4.figure.subplots()
        self.grafico4.set_xlabel('Jump Height (nm)', size=8)
        self.grafico4.set_ylabel('Unfolding Force (pN)', size=8)
        self.grafico4.set_title('Force-Extension Correlation', size=9, fontweight='bold')

    def _connect_signals(self):
        self.Open_JSON.clicked.connect(self.file_open)
        self.Plot_button.clicked.connect(self.plot_pulse)
        self.Histo.clicked.connect(self.do_histogram)
        self.Save_file.clicked.connect(self.save_to_file)
        self.Transition_detect.clicked.connect(self.analyze_all_pulses)

    def file_open(self):
        """Open JSON file dialog"""
        name = QtWidgets.QFileDialog.getOpenFileName(self, "Open JSON File")
        self.file_path = name[0]
        self._invalidate_cache()
        return self.file_path

    def _invalidate_cache(self):
        """Clear cached data when new file is loaded"""
        self.data = None
        self.xs_cache = None
        self.forces_cache = None
        self.times_cache = None
        self.xs_smth_cache = None

    def _load_and_cache_data(self):
        """Load JSON and cache all processed data
        
        Uses checkBoth to determine data loading:
        - Unchecked: Load all pulses (unfolding only)
        - Checked: Load every 2nd pulse (dual process: folding + unfolding)
        """
        if self.xs_cache is not None:
            return

        with open(self.file_path, 'r') as f:
            self.data = json.load(f)

        xs, forces, times = [], [], []
        step = 2 if self.checkBoth.isChecked() else 1  # Dual event mode

        for i in range(0, len(self.data), step):
            pulse_data = self.data[f"Pulse_Number_{i}"]
            z = np.array(pulse_data["z"])
            F  =np.array(pulse_data["force"])
            I = np.array(pulse_data["current"])
            t = np.array(pulse_data["time"])
            
            xs.append(z)
            forces.append(F)
            times.append(t)

        # Cache everything including smoothed data
        self.xs_cache = xs
        self.forces_cache = forces
        self.times_cache = times
        self.xs_smth_cache = [savgol_filter(x, 51, 4) for x in xs]
        
        # Debug: check force range
        all_forces = np.concatenate(forces)
        print(f"Force range after loading: {np.min(all_forces):.2f} to {np.max(all_forces):.2f} pN")
        print(f"Mean force: {np.mean(all_forces):.2f} pN")
        
        self.Output.setText(f"Pulses loaded: {len(xs)}")

    def plot_pulse(self):
        """Plot selected pulse"""
        self._load_and_cache_data()
        i = self.Pulse_num.value()

        self.grafico.clear()
        self.grafico.set_ylabel('Force (pN)', size=6)
        self.grafico.set_xlabel('Extension (nm)', size=6)
        
        self.grafico.plot(self.xs_cache[i], self.forces_cache[i], alpha=1, lw=1)
        self.grafico.plot(self.xs_smth_cache[i], self.forces_cache[i], alpha=1, lw=1)
        self.grafico.tick_params(axis='both', labelsize=6)
        self.grafico.figure.canvas.draw()

    def do_histogram(self):
        """Calculate unfolding forces and plot histogram with Bell-Evans fit"""
        self._load_and_cache_data()
        threshold = self.Thr.value()
        
        self.unfolding_forces = []
        pulse_idx = []

        for i, xs_smth in enumerate(self.xs_smth_cache):
            jump_i = detect_jump(xs_smth, threshold)
            if jump_i is not None:
                self.unfolding_forces.append(self.forces_cache[i][jump_i])
                pulse_idx.append(i)

        print(f"Unfolding force range: {min(self.unfolding_forces):.2f} to {max(self.unfolding_forces):.2f} pN")
        print(f"Mean unfolding force: {np.mean(self.unfolding_forces):.2f} pN")
        print(f"Number of events: {len(self.unfolding_forces)}")

        # External plot
        plt.figure(5)
        plt.plot(pulse_idx, self.unfolding_forces, marker='o', linestyle="-", lw=1, markersize=2)
        plt.title("Unfolding Forces (index)")
        plt.xlabel("Pulse Index")
        plt.ylabel("Force (pN)")
        plt.show()

        # Histogram with fit
        self.grafico3.clear()
        self.grafico3.set_xlabel('Unfolding Force (pN)', size=6)
        self.grafico3.set_ylabel('Counts', size=6)
        self.grafico3.tick_params(axis='both', labelsize=6)

        r = self.r_box.value()
        print(f"Pulling rate (r): {r} pN/s")
        
        try:
            x_beta, k0, x_fit, y_fit, _, _ = fit_bell_evans_model(self.unfolding_forces, r)
            
            print(f"Fit results: x‡ = {x_beta:.2f} nm, k0 = {k0:.2e} /s")
            
            self.grafico3.hist(self.unfolding_forces, bins=30, color='lightblue', 
                              alpha=0.5, edgecolor='black', density=True)
            self.grafico3.plot(x_fit, y_fit, color='red', lw=1, label='Bell-Evans fit')
            
            self.xdag_label.setText(f"Estimate x‡: {x_beta:.2f} nm")
            self.k0_label.setText(f"k0 estimated: {k0:.2e} /s")
            
        except Exception as e:
            print(f"Fit failed: {e}")
            import traceback
            traceback.print_exc()
            self.grafico3.hist(self.unfolding_forces, bins=30, color='lightblue', 
                              alpha=0.5, edgecolor='black', density=True)
            self.xdag_label.setText("Fit failed")
            self.k0_label.setText("Check data")
        
        self.grafico3.figure.canvas.draw()
        self.Output.setText(f"Ramps: {len(self.data)}, Events: {len(self.unfolding_forces)}")

        return self.unfolding_forces

    def _find_transition(self, xs_list, time_list, window=91, polyorder=4, 
                        search_range=[0.05, 0.99], local_window=15):
        """Detect fold-unfold transition point
        
        Filters out ramps with insufficient extension (< threshold - 10 nm).
        For Talin R3: minimum ~20 nm jump required for valid unfolding event.
        """
        n_points = len(xs_list)
        window = min(window, n_points)
        if window < polyorder + 1:
            polyorder = max(1, window - 1)

        smoothed = savgol_filter(xs_list, window, polyorder, mode='nearest')

        # Find transition in search range
        start_idx = int(n_points * search_range[0])
        end_idx = int(n_points * search_range[1])
        derivative = np.diff(smoothed[start_idx:end_idx])
        
        # Average of top 5 derivative points
        top5_indices = np.argsort(np.abs(derivative))[-5:]
        max_derivative_idx = int(np.mean(top5_indices)) + start_idx
        transition_point = time_list[max_derivative_idx]

        # Local fold/unfold levels
        kernel = np.ones(6) / 6
        roll_mean = np.convolve(smoothed, kernel, mode='same')

        left_slice = slice(max(0, max_derivative_idx - local_window), max_derivative_idx)
        right_slice = slice(max_derivative_idx + 1, 
                           min(n_points, max_derivative_idx + local_window))

        fold_region_end = left_slice.start + np.argmin(roll_mean[left_slice])
        unfold_region_start = right_slice.start + np.argmax(roll_mean[right_slice])

        jump_height = smoothed[unfold_region_start] - smoothed[fold_region_end]
        min_jump = self.Thr.value() - 10  # Talin R3 unfolding length filter
        
        if jump_height < min_jump:
            return None, False

        return (jump_height, transition_point, fold_region_end, 
                unfold_region_start, max_derivative_idx), True

    def analyze_all_pulses(self):
        """Analyze all pulses for transitions
        
        Transition Detection filters ramps that don't reach the minimum extension 
        threshold (typically 20 nm for Talin R3 unfolding). Only valid unfolding 
        events with sufficient extension are included in the analysis.
        
        Plots only first 10 valid transitions for clarity.
        """
        self._load_and_cache_data()
        threshold = self.Thr.value()
        
        results = []
        jump_heights = []
        unfolding_forces = []
        valid_pulse_indices = []
        
        plt.figure(1, figsize=(12, 6))
        plt.clf()
        
        plot_count = 0
        max_plots = 10  # Show only first 10 for clarity
        
        for i, xs_smth in enumerate(self.xs_smth_cache):
            jump_i = detect_jump(xs_smth, threshold)
            if jump_i is None:
                continue

            smoothed = savgol_filter(self.xs_cache[i], 91, 4, mode='nearest')
            result, valid = self._find_transition(self.xs_cache[i], self.times_cache[i])

            if not valid:
                print(f"Pulse {i} rejected: insufficient jump (< 20 nm)")
                continue

            jump_height, transition_point, fold_end, unfold_start, max_idx = result
            unfolding_force = self.forces_cache[i][jump_i]

            results.append({
                "pulse_index": i,
                "jump_height": jump_height,
                "transition_point": transition_point,
                "unfolding_force": unfolding_force
            })
            jump_heights.append(jump_height)
            unfolding_forces.append(unfolding_force)
            valid_pulse_indices.append(i)

            # Plot only first 10 transitions for visibility
            if plot_count < max_plots:
                plt.subplot(2, 5, plot_count + 1)
                plt.plot(self.times_cache[i], smoothed, color="orange", alpha=0.8, lw=1.5)
                plt.scatter(self.times_cache[i][fold_end], smoothed[fold_end], 
                           color="blue", zorder=5, s=25, label='Fold')
                plt.scatter(self.times_cache[i][unfold_start], smoothed[unfold_start], 
                           color="purple", zorder=5, s=25, label='Unfold')
                plt.title(f"Pulse {i}", fontsize=8)
                plt.xlabel("Time (s)", fontsize=7)
                plt.ylabel("Extension (nm)", fontsize=7)
                plt.tick_params(labelsize=6)
                plt.grid(True, alpha=0.3)
                if plot_count == 0:
                    plt.legend(fontsize=6)
                plot_count += 1

        plt.suptitle(f"Transition Detection - First {min(plot_count, max_plots)} Valid Events", 
                    fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Jump height distribution
        plt.figure(2, figsize=(8, 5))
        plt.hist(jump_heights, bins=35, density=True, edgecolor="darkgray", 
                alpha=0.7, color='lightblue', label=f"Mean: {np.mean(jump_heights):.1f} nm")
        plt.xlim(0, 50)
        plt.xlabel("Extension (nm)", fontsize=10)
        plt.ylabel("Density", fontsize=10)
        plt.title("Jump Height Distribution", fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Update canvas with Force vs Jump Height correlation
        self.grafico4.clear()
        self.grafico4.scatter(jump_heights, unfolding_forces, 
                             color='steelblue', alpha=0.6, s=40, edgecolors='navy', linewidths=0.8)
        
        # Add trend line if enough data
        if len(jump_heights) > 3:
            from scipy.stats import linregress
            slope, intercept, r_value, _, _ = linregress(jump_heights, unfolding_forces)
            x_trend = np.linspace(min(jump_heights), max(jump_heights), 100)
            y_trend = slope * x_trend + intercept
            self.grafico4.plot(x_trend, y_trend, 'r--', lw=1.5, alpha=0.7)
        
        self.grafico4.set_xlabel('Jump Height (nm)', size=8)
        self.grafico4.set_ylabel('Unfolding Force (pN)', size=8)
        self.grafico4.set_title(f'Force-Extension Correlation (n={len(valid_pulse_indices)})', 
                               size=9, fontweight='bold')
        self.grafico4.tick_params(axis='both', labelsize=7)
        self.grafico4.legend(fontsize=7)
        self.grafico4.grid(True, alpha=0.3)
        self.grafico4.figure.canvas.draw()

        print(f"\n=== Analysis Summary ===")
        print(f"Valid events: {len(results)}")
        print(f"Mean jump height: {np.mean(jump_heights):.2f} ± {np.std(jump_heights):.2f} nm")
        print(f"Mean unfolding force: {np.mean(unfolding_forces):.2f} ± {np.std(unfolding_forces):.2f} pN")

        return results, jump_heights

    def save_to_file(self):
        """Save unfolding forces to CSV"""
        if not self.unfolding_forces:
            self.Output.setText("No data to save!")
            return
        np.savetxt("Unfolding_forces.csv", self.unfolding_forces, delimiter=",")
        self.Output.setText("Data saved to Unfolding_forces.csv")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
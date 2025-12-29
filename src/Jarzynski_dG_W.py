"""
JARZYNSKI WORK ANALYZER - Non-Equilibrium Free Energy Estimation

Analyzes magnetic tweezers data using Jarzynski equality to calculate free energy
changes from irreversible work measurements in single-molecule experiments.

Input: JSON file with pulse data (z, force, time, current)
Output: Work distributions, WLC fits, free energy estimates (ΔG)

Key Features:
- Total work calculation from force-extension curves
- WLC (Worm-Like Chain) model fitting for polymer stretching
- FJC (Freely-Jointed Chain) model for orientation work
- Work corrections for elastic contributions
- Jarzynski equality: ΔG = -kT ln⟨e^(-W/kT)⟩
- Jackknife error estimation
"""

import numpy as np
import pandas as pd
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
from scipy.optimize import curve_fit, minimize, fsolve
from scipy.integrate import simpson, quad
from scipy.stats import sem


UI_FILE = "UI_Jarzynski.ui"
Ui_MainWindow, QtBaseClass = loadUiType(UI_FILE)

KBT = 4.11  # pN·nm at room temperature

def wlc_force(x, Lc, Lp, L0=0):
    """Worm-Like Chain force model"""
    return (KBT / Lp) * (0.25 * (1 - (x - L0) / Lc)**(-2) + (x - L0) / Lc - 0.25)


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self._configure_style()
        self._init_state()
        self._setup_plots()
        self._connect_signals()
        self._set_default_values()

    def _configure_style(self):
        """Configure matplotlib and PyQt dark theme"""
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
        """Initialize cached data and state variables"""
        self.file_path = ""
        self.xs_cache = None
        self.forces_cache = None
        self.times_cache = None
        self.xs_smooth_cache = None
        self.data = None
        
        # Analysis results cache
        self.x_avg_cache = None
        self.f_avg_cache = None
        self.work_list_cache = None
        self.wlc_params_cache = None

    def _setup_plots(self):
        """Setup plotting canvases"""
        # Force-Extension plot
        canvas = FigureCanvas(Figure(figsize=(8, 6), constrained_layout=True,
                                    edgecolor='black', frameon=True))
        self.Plotting_box.addWidget(canvas)
        self.grafico = canvas.figure.subplots()
        self.grafico.set_xlabel('Extension (nm)', size=8)
        self.grafico.set_ylabel('Force (pN)', size=8)

        # Work histogram
        canvas_hist = FigureCanvas(Figure(figsize=(8, 6), constrained_layout=True))
        self.W_hist.layout().addWidget(canvas_hist)
        self.grafico_hist = canvas_hist.figure.subplots()

    def _connect_signals(self):
        """Connect UI buttons to methods"""
        self.Open_JSON.clicked.connect(self.file_open)
        self.Plot_button.clicked.connect(self.plot_pulse)
        self.Plot_avg_traj.clicked.connect(self.plot_average_trajectory)
        self.compute_sgW.clicked.connect(self.compute_total_work)
        self.global_fit.clicked.connect(self.fit_wlc_global)
        self.Wp_strech.clicked.connect(self.compute_w_protein)
        self.Wh_strech.clicked.connect(self.compute_w_handles)
        self.FJC_W.clicked.connect(self.compute_fjc_work)
        self.clean_w.clicked.connect(self.compute_clean_work)
        self.dG_distribution.clicked.connect(self.compute_free_energy)
        self.Save_file.clicked.connect(self.save_results)

    def _set_default_values(self):
        """Set default GUI values"""
        self.Fmin_2.setValue(4.5)
        self.Fmin.setValue(12)
        self.Fmax_2.setValue(7.8)
        self.Fmax.setValue(18)
        self.Guess_4.setValue(74.0)
        self.Guess_5.setValue(0.6)
        self.Guess_1.setValue(110.0)
        self.Guess_2.setValue(0.4)
        self.Guess_3.setValue(0.0)
        self.percentile.setValue(1.0)
        self.Lc_fjc.setValue(7)
        self.Lk_fjc.setValue(1)
        self.lower_limit.setValue(5)
        self.upper_limit.setValue(18)

    # ========== FILE I/O ==========

    def file_open(self):
        """Open JSON file dialog"""
        name = QtWidgets.QFileDialog.getOpenFileName(self, "Open JSON File")
        self.file_path = name[0]
        self._invalidate_cache()
        return self.file_path

    def _invalidate_cache(self):
        """Clear all cached data when new file is loaded"""
        self.xs_cache = None
        self.forces_cache = None
        self.times_cache = None
        self.xs_smooth_cache = None
        self.x_avg_cache = None
        self.f_avg_cache = None
        self.work_list_cache = None
        self.wlc_params_cache = None

    def _load_and_cache_data(self):
        """Load JSON and cache all processed data"""
        if self.xs_cache is not None:
            return

        with open(self.file_path, 'r') as f:
            self.data = json.load(f)

        xs, forces, times = [], [], []
        
        for i in range(len(self.data)):
            pulse_data = self.data[f"Pulse_Number_{i}"]
            z = np.array(pulse_data["z"])
            force_array = np.abs(np.array(pulse_data["force"]))
            t = np.array(pulse_data["time"]) - pulse_data["time"][0]
            I = np.array(pulse_data["current"])
            F = (1.504e-5)* I**2 - 0.0133*I

            xs.append(z)
            forces.append(F)
            times.append(t)

        # Pre-smooth all data
        self.xs_cache = xs
        self.forces_cache = forces
        self.times_cache = times
        self.xs_smooth_cache = [savgol_filter(x, 51, 4) for x in xs]
        
        print(f"Loaded {len(xs)} pulses")

    # ========== PLOTTING ==========

    def plot_pulse(self):
        """Plot individual pulse"""
        self._load_and_cache_data()
        i = self.Pulse_num.value()

        self.grafico.clear()
        self.grafico.set_xlabel('Extension (nm)', size=6)
        self.grafico.set_ylabel('Force (pN)', size=6)
        
        self.grafico.plot(self.xs_cache[i], self.forces_cache[i], alpha=0.5, label='Raw')
        self.grafico.plot(self.xs_smooth_cache[i], self.forces_cache[i], 
                         alpha=0.8, label='Smoothed')
        self.grafico.legend(fontsize=7)
        self.grafico.figure.canvas.draw()

    def plot_average_trajectory(self):
        """Compute and plot average force-extension curve"""
        self._compute_average_trajectory()

        self.grafico.clear()
        self.grafico.set_xlabel('Extension (nm)', size=8)
        self.grafico.set_ylabel('Force (pN)', size=8)
        self.grafico.plot(self.x_avg_cache, self.f_avg_cache, 
                         '-', lw=1.5, alpha=0.8, color='orange', label='Average')
        self.grafico.legend(fontsize=5)
        self.grafico.figure.canvas.draw()

    def _compute_average_trajectory(self):
        """Compute average trajectory (cached)"""
        if self.x_avg_cache is not None:
            return

        self._load_and_cache_data()
        n = len(self.xs_cache)
        min_len = min(len(self.xs_cache[i]) for i in range(n))
        
        xs_array = np.array([self.xs_cache[i][:min_len] for i in range(n)])
        fs_array = np.array([self.forces_cache[i][:min_len] for i in range(n)])
        
        self.x_avg_cache = np.mean(xs_array, axis=0)
        self.f_avg_cache = np.mean(fs_array, axis=0)

    # ========== WORK CALCULATION ==========

    def compute_total_work(self):
        """Calculate total work for all pulses using trapezoidal integration"""

        self._load_and_cache_data()
        work_list = []
        f_lower = self.lower_limit.value()
        f_upper = self.upper_limit.value()

        for i in range(len(self.forces_cache)):
            try:
                forces = self.forces_cache[i]
                
                idx_min = next(j for j in range(1, len(forces))
                              if forces[j] > f_lower and forces[j-1] < f_lower)
                
                idx_max = next(j for j in range(1, len(forces))
                              if forces[j] > f_upper and forces[j-1] < f_upper)
                
                F_slice = forces[idx_min:idx_max]
                x_slice = self.xs_smooth_cache[i][idx_min:idx_max]
                
                work = np.trapz(np.abs(F_slice), x_slice)
                work_list.append(work)
                
            except StopIteration:
                print(f"Pulse {i}: skipped (limits not reached)")
                continue

        self.work_list_cache = work_list        
        self._plot_work_histogram()
        return work_list

    def _plot_work_histogram(self):
        """Plot work distribution histogram"""
        self.grafico_hist.clear()
        self.grafico_hist.set_xlabel('Total Work [pN·nm]', size=8)
        self.grafico_hist.set_ylabel('Probablity Density', size=8)
        self.grafico_hist.hist(self.work_list_cache, bins=200, 
                              edgecolor='black', alpha=0.6, color='orange')
        self.grafico_hist.figure.canvas.draw()

    # ========== WLC FITTING ==========

    def fit_wlc_global(self):
        """Global WLC fit for both handles and protein"""

        self._compute_average_trajectory()
        
        fmin_dx, fmax_dx = self.Fmin.value(), self.Fmax.value()
        fmin_sx, fmax_sx = self.Fmin_2.value(), self.Fmax_2.value()

        mask_dx = (self.f_avg_cache >= fmin_dx) & (self.f_avg_cache <= fmax_dx)
        x_dx, f_dx = self.x_avg_cache[mask_dx], self.f_avg_cache[mask_dx]
    
        mask_sx = (self.f_avg_cache >= fmin_sx) & (self.f_avg_cache <= fmax_sx)
        x_sx, f_sx = self.x_avg_cache[mask_sx], self.f_avg_cache[mask_sx]

        # Global optimization
        def cost_function(params):
            Lc_dx, Lp_dx, Lc_sx, Lp_sx, L0 = params
            try:
                pred_dx = wlc_force(x_dx, Lc_dx, Lp_dx, L0)
                pred_sx = wlc_force(x_sx, Lc_sx, Lp_sx, L0)
                mse = np.mean((f_dx - pred_dx)**2) + np.mean((f_sx - pred_sx)**2)
                return mse
            except:
                return 1e10

        guess = [self.Guess_1.value(), self.Guess_2.value(), 
                self.Guess_4.value(), self.Guess_5.value(), self.Guess_3.value()]
        bounds = [(50, 190), (0.3, 0.7), (0, 140), (0.3, 0.7), (-30, 50)]

        result = minimize(cost_function, guess, method='L-BFGS-B', bounds=bounds)

        if result.success:
            Lc_dx, Lp_dx, Lc_sx, Lp_sx, L0 = result.x
            self.wlc_params_cache = (Lc_dx, Lp_dx, Lc_sx, Lp_sx, L0)
            
            self.Label_WLCfit_dx.setText(f"Lc_dx={Lc_dx:.2f} Lp_dx={Lp_dx:.2f}")
            self.Label_WLCfit_sx.setText(f"Lc_sx={Lc_sx:.2f} Lp_sx={Lp_sx:.2f} L0={L0:.2f}")
            
            self.grafico.clear()
            self.grafico.plot(self.x_avg_cache, self.f_avg_cache, 
                              '-', lw=1.2, color='orange', alpha=0.4, label='Average')
            
            mask_plot_dx = (self.f_avg_cache >= fmin_dx) & (self.f_avg_cache <= fmax_dx)
            x_plot_dx = self.x_avg_cache[mask_plot_dx]
            
            if len(x_plot_dx) > 0:
                y_plot_dx = wlc_force(x_plot_dx, Lc_dx, Lp_dx, L0)
                self.grafico.plot(x_plot_dx, y_plot_dx, 
                                 '--', lw=1.5, color='orange', label='WLC fit (right)')

            mask_plot_sx = (self.f_avg_cache >= fmin_sx) & (self.f_avg_cache <= fmax_sx)
            x_plot_sx = self.x_avg_cache[mask_plot_sx]
            
            if len(x_plot_sx) > 0:
                y_plot_sx = wlc_force(x_plot_sx, Lc_sx, Lp_sx, L0)
                self.grafico.plot(x_plot_sx, y_plot_sx, 
                                 '--', lw=1.5, color='purple', label='WLC fit (left)')

            self.grafico.legend(fontsize=7)
            self.grafico.set_ylim(self.f_avg_cache.min() - 1, self.f_avg_cache.max() + 2)
            self.grafico.figure.canvas.draw()
        else:
            print("Global fit failed:", result.message)

        return self.wlc_params_cache

    def compute_wlc_work(self):
        """Compute work from WLC models"""
        Lc_dx, Lp_dx, Lc_sx, Lp_sx, L0 = self.fit_wlc_global()
        
        self._compute_average_trajectory()
        f_lower = self.lower_limit.value()
        f_upper = self.upper_limit.value()
        
        idx_min = np.where(self.f_avg_cache > f_lower)[0][0]
        idx_max = np.where(self.f_avg_cache > f_upper)[0][0]
        xmin = self.x_avg_cache[idx_min]
        xmax = self.x_avg_cache[idx_max]

        x_dx = np.linspace(xmin, xmax, 1000)
        f_dx = wlc_force(x_dx, Lc_dx, Lp_dx, L0)
        work_dx = simpson(f_dx, x_dx)

        x_sx = np.linspace(-5, xmin, 1000)
        f_sx = wlc_force(x_sx, Lc_sx, Lp_sx, L0)
        f_sx = f_sx[f_sx > 0.1]  
        work_sx = simpson(f_sx, x_sx[:len(f_sx)])

        self.Label_WorkWLC_dx.setText(f"Work WLC dx={work_dx:.2f}")
        self.Label_WorkWLC_sx.setText(f"Work WLC sx={work_sx:.2f}")

        return work_dx, work_sx

    def compute_w_protein(self):
        """Calculate elastic work for protein stretching"""
        Lc_dx, Lp_dx, Lc_sx, Lp_sx, L0 = self.fit_wlc_global()
        Lc = Lc_dx - Lc_sx
        Lp = Lp_dx
        fmax = self.upper_limit.value()

        xmin = fsolve(lambda x: wlc_force(x, Lc, Lp, 0) - 0, 0.5 * Lc)[0]
        xmax = fsolve(lambda x: wlc_force(x, Lc, Lp, 0) - fmax, 0.5 * Lc)[0]
        
        Ws_p, _ = quad(lambda x: wlc_force(x, Lc, Lp, 0), xmin, xmax)
        print(f"W_protein: {Ws_p:.2f} pN·nm")
        
        return Ws_p

    def compute_w_handles(self):
        """Calculate elastic work for handle stretching"""
        Lc_dx, Lp_dx, Lc_sx, Lp_sx, L0 = self.fit_wlc_global()
        Lc = Lc_sx - L0 - 7
        Lp = Lp_sx
        fmin, fmax = self.lower_limit.value(), self.upper_limit.value()

        xmin = fsolve(lambda x: wlc_force(x, Lc, Lp, L0) - fmin, 0.5 * Lc + L0)[0]
        xmax = fsolve(lambda x: wlc_force(x, Lc, Lp, L0) - fmax, 0.5 * Lc + L0)[0]
        
        Ws_h, _ = quad(lambda x: wlc_force(x, Lc, Lp, L0), xmin, xmax)
        print(f"W_handles: {Ws_h:.2f} pN·nm")
        
        return Ws_h

    # ========== FJC MODEL ==========

    def compute_fjc_work(self):
        """Calculate orientation work using FJC model"""
        Lc = self.Lc_fjc.value()
        lk = self.Lk_fjc.value()
        f0 = self.lower_limit.value()
        fmax = self.upper_limit.value()

        F = np.linspace(0.01, 25, 1000)
        beta = F * lk / KBT
        fjc_x = Lc * (1 / np.tanh(beta) - 1 / beta)

        mask1 = F <= f0
        W_orient = simpson(fjc_x[mask1][-1] - fjc_x[mask1], F[mask1])

        mask2 = (F >= f0) & (F <= fmax)
        W_stretch = simpson(fjc_x[mask2][-1] - fjc_x[mask2], F[mask2])

        self.Label_fjc.setText(f"W_orient={W_orient:.2f} pN·nm")
        print(f"W_orientation: {W_orient:.2f}, W_stretch: {W_stretch:.2f}")

        return W_orient, W_stretch

    # ========== WORK CORRECTION ==========

    def compute_clean_work(self):
        """Subtract elastic contributions from total work"""
        work_list = self.compute_total_work()
        Ws_h = self.compute_w_handles()
        Ws_p = self.compute_w_protein()
        W_orient, W_stretch = self.compute_fjc_work()

        correction = Ws_h + Ws_p - W_stretch - W_orient
        print(f"Work correction: {correction:.2f} pN·nm")

        work_cleaned = np.array(work_list) - correction

        low_perc = np.percentile(work_cleaned, self.percentile.value())
        #work_cleaned_filtered = work_cleaned[work_cleaned > low_perc]
        if self.percentile.value() <= 0:
            work_cleaned_filtered = work_cleaned
        else:
            work_cleaned_filtered = work_cleaned[work_cleaned >= low_perc]

        print(f"Cleaned work: mean={np.mean(work_cleaned_filtered):.2f} ± "
              f"{sem(work_cleaned_filtered):.2f} pN·nm")

        plt.figure(figsize=(10, 5))
        plt.hist(work_cleaned, bins=130, alpha=0.5, edgecolor="black", 
                density=True, label=f"All (mean={np.mean(work_cleaned):.1f})")
        plt.hist(work_cleaned_filtered, bins=130, alpha=0.7, edgecolor="black",
                density=True, label=f"Filtered (mean={np.mean(work_cleaned_filtered):.1f})")
        plt.xlabel("Unfolding Work [pN·nm]")
        plt.ylabel("Density")
        plt.title("Work Distribution After Elastic Correction")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return work_cleaned, work_cleaned_filtered

    # ========== FREE ENERGY ==========

    def compute_free_energy(self):
        """Calculate free energy using Jarzynski equality with Jackknife error"""
        _, work_cleaned = self.compute_clean_work()
        
        # Jarzynski equality: ΔG = -kT ln⟨e^(-W/kT)⟩
        dG = -KBT * np.log(np.mean(np.exp(-work_cleaned / KBT)))
        print(f"\n=== FREE ENERGY (Jarzynski) ===")
        print(f"ΔG = {dG:.2f} pN·nm")

        # Jackknife error estimation
        n = len(work_cleaned)
        jk_estimates = []
        
        for i in range(n):
            data_jk = np.delete(work_cleaned, i)
            dG_jk = -KBT * np.log(np.mean(np.exp(-data_jk / KBT)))
            jk_estimates.append(dG_jk)

        jk_mean = np.mean(jk_estimates)
        jk_var = ((n - 1) / n) * np.sum((jk_estimates - jk_mean)**2)
        jk_std = np.sqrt(jk_var)

        print(f"Jackknife: ΔG = {jk_mean:.2f} ± {jk_std:.2f} pN·nm")
        print(f"Standard error: {jk_std:.2f} pN·nm")

        outcome = f"ΔG = {jk_mean:.2f} ± {jk_std:.2f} pN·nm"
        self.Label_dG.setText(outcome)
        return dG, jk_std

    # ========== FILE SAVING ==========

    def save_results(self):
        """Save work analysis results to CSV"""
        work_list = self.work_list_cache
        if work_list is None:
            print("No work data to save!")
            return

        work_cleaned, work_filtered = self.compute_clean_work()
        
        df = pd.DataFrame({
            "Pulse": list(range(len(work_list))),
            "Work_total": work_list,
            "Work_cleaned": work_cleaned[:len(work_list)]
        })
        
        df.to_csv("work_results.csv", index=False)
        print("✓ Results saved to work_results.csv")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
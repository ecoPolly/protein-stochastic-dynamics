#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pulse Analyzer - Constant Force Analysis Tool
Analyzes protein folding/unfolding dynamics from force spectroscopy data

Author: w2040021
Created: Mon Feb 3 11:16:42 2025
"""

import numpy as np
import sys
import json
import os
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
from numpy.fft import fftshift
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from hmmlearn import hmm
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QFileDialog, QMessageBox)
from PyQt5.uic import loadUiType
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================
DEFAULT_SMOOTHING_WINDOW = 51
DEFAULT_DIFF_COEFF = 3000  # nm²/s
DEFAULT_HMM_STATES = 2
PEAK_PROMINENCE = 0.3
PEAK_WIDTH = 6
PEAK_DISTANCE = 6
EPSILON = 1e-12  # Small value to avoid log(0)

# ============================================================================
# GUI CONFIGURATION
# ============================================================================
UI_FILE = "UI_constant_force.ui"
Ui_MainWindow, QtBaseClass = loadUiType(UI_FILE)

# ============================================================================
# MATPLOTLIB STYLING
# ============================================================================
PLOT_PARAMS = {
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
}

DARK_STYLESHEET = """
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


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    """
    Main application window for Pulse Analyzer
    
    Provides GUI interface for analyzing protein folding/unfolding dynamics
    including potential energy landscape reconstruction, HMM state analysis,
    and kinetic rate calculations.
    """
    
    def __init__(self):
        """Initialize the application window and setup UI components"""
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Pulse ANALYZER')
        
        # CHANGED: Initialize data storage attributes at startup
        self.file_path = ""
        self.t = None  # Time array
        self.x = None  # Raw extension data
        self.x_smth = None  # Smoothed extension data
        self.counts = None  # Histogram counts for U(x)
        self.bin_centers = None  # Histogram bin centers
        self.U_x = None  # Potential energy landscape
        self.bin_centers_dec = None  # Deconvolved bin centers
        self.U_dec_shifted = None  # Deconvolved potential
        self.dwell_times_list = []  # List of dwell times from HMM
        self.states = None  # HMM state sequence
        
        # CHANGED: Setup plot styling
        self._setup_plot_style()
        
        # CHANGED: Connect GUI signals to slots
        self._connect_signals()
        
        # CHANGED: Initialize plot canvases
        self._setup_plot_canvases()
        
        # CHANGED: Set default values
        self._set_default_values()
    
    def _setup_plot_style(self):
        """Configure matplotlib and Qt styling"""
        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
        plt.style.use('dark_background')
        plt.rcParams.update(PLOT_PARAMS)
        app.setStyleSheet(DARK_STYLESHEET)
    
    def _connect_signals(self):
        """Connect GUI button signals to their handler methods"""
        # CHANGED: Organized signal connections
        self.Open_JSON.clicked.connect(self.file_open)
        self.Plot_button.clicked.connect(self.plotter)
        self.Plot_Ux_button.clicked.connect(self.plot_ux)
        self.btn_dwell.clicked.connect(self.dwell_analysis)
        self.HMM_rates.clicked.connect(self.hmm_analysis)
        self.deconvolution_button.clicked.connect(self.deconvolve_ux)
        self.btn_update.clicked.connect(self.plot_subportion)
        self.save_dwell.clicked.connect(self.save_dwell_times)
    
    def _setup_plot_canvases(self):
        """Initialize all matplotlib canvas widgets"""
        # CHANGED: Main trajectory plot
        main_canvas = FigureCanvas(Figure(figsize=(8, 6), constrained_layout=True))
        self.Plotting_box.addWidget(main_canvas)
        self.grafico = main_canvas.figure.subplots()
        self.grafico.set_xlabel('Time (s)', size=8)
        self.grafico.set_ylabel('Extension (nm)', size=8)
        x_init = np.linspace(0, 0.03, 100)
        self.grafico.plot(x_init, 0 * x_init)
        
        # CHANGED: Dwell time histograms
        self.dwell_down_canvas = FigureCanvas(Figure())
        self.dwell_up_canvas = FigureCanvas(Figure())
        
        lay_dwell_down = QVBoxLayout(self.widget_dwell_down)
        lay_dwell_down.setContentsMargins(0, 0, 0, 0)
        lay_dwell_down.addWidget(self.dwell_down_canvas)
        
        lay_dwell_up = QVBoxLayout(self.widget_dwell_up)
        lay_dwell_up.setContentsMargins(0, 0, 0, 0)
        lay_dwell_up.addWidget(self.dwell_up_canvas)
        
        # CHANGED: HMM states visualization
        self.hmm_states_layout = QtWidgets.QVBoxLayout(self.hmm_states)
        self.hmm_states_canvas = FigureCanvas(Figure())
        self.hmm_states_layout.addWidget(self.hmm_states_canvas)
        self.hmm_states = self.hmm_states_canvas
    
    def _set_default_values(self):
        """Set default values for GUI input fields"""
        # CHANGED: Centralized default value setting
        self.smooth.setValue(DEFAULT_SMOOTHING_WINDOW)
        self.diff_coeff.setValue(DEFAULT_DIFF_COEFF)
    
    def file_open(self):
        """
        Open file dialog to select JSON data file
        
        Returns:
            str: Path to selected file
        """
        # CHANGED: Renamed from File_open, updated to use instance variable
        name = QtWidgets.QFileDialog.getOpenFileName(self, "Open JSON File")
        self.file_path = name[0]
        return self.file_path
    
    def load_json(self):
        """
        Load pulse data from JSON file
        
        Returns:
            tuple: (extensions, forces, times) - Arrays of measurement data
        """
        # CHANGED: Renamed from Load_JSON, improved error handling
        if not self.file_path:
            QMessageBox.warning(self, "Error", "No file selected. Please open a JSON file first.")
            return None, None, None
        
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load JSON: {e}")
            return None, None, None
        
        pulse_num = self.Pulse_num.value()
        pulse_key = f"Pulse_Number_{pulse_num}"
        
        if pulse_key not in data:
            QMessageBox.warning(self, "Error", f"Pulse {pulse_num} not found in data")
            return None, None, None
        
        # CHANGED: More explicit data extraction
        xs = [np.array(data[pulse_key]["z"])]
        forces = [np.array(data[pulse_key]["force"])]
        times = [np.array(data[pulse_key]["time"]) - data[pulse_key]["time"][0]]
        
        return xs, forces, times
    
    def plotter(self):
        """
        Plot raw and smoothed trajectory data
        
        Displays time vs extension plot with both raw and filtered data
        """
        # CHANGED: Renamed from Plotter, improved documentation
        xs, forces, times = self.load_json()
        
        if xs is None:
            return
        
        pulse_num = self.Pulse_num.value()
        smoothing_window = self.smooth.value()
        
        # CHANGED: Direct smoothing instead of unnecessary list
        xs_smth = savgol_filter(xs[0], smoothing_window, 4)
        
        # CHANGED: Store as instance variables for other methods
        self.t = times[0]
        self.x = xs[0]
        self.x_smth = xs_smth
        
        # CHANGED: Plot in GUI
        self.grafico.clear()
        self.grafico.set_xlabel('Time (s)', size=6)
        self.grafico.set_ylabel('Extension (nm)', size=6)
        self.grafico.plot(self.t, self.x, lw=0.6, label='Raw', alpha=0.5)
        self.grafico.plot(self.t, self.x_smth, lw=0.6, label='Smoothed')
        self.grafico.legend(fontsize=6)
        self.grafico.figure.canvas.draw()
    
    def plot_subportion(self):
        """
        Plot zoomed-in portion of trajectory based on time limits
        
        Uses time range from GUI input fields (edit_tmin, edit_tmax)
        """
        # CHANGED: Added validation check
        if self.t is None or self.x is None:
            QMessageBox.warning(self, "Error", "No data loaded. Please plot data first.")
            return
        
        # CHANGED: Better error handling for time range input
        try:
            tmin = float(self.edit_tmin.text())
        except (ValueError, AttributeError):
            tmin = np.min(self.t)
        
        try:
            tmax = float(self.edit_tmax.text())
        except (ValueError, AttributeError):
            tmax = np.max(self.t)
        
        mask = (self.t >= tmin) & (self.t <= tmax)
        t_plot = self.t[mask]
        x_plot = self.x[mask]
        x_smth_plot = self.x_smth[mask]
        
        # CHANGED: Plot sub-portion
        self.grafico.clear()
        self.grafico.set_xlabel('Time (s)', size=8)
        self.grafico.set_ylabel('Extension (nm)', size=8)
        self.grafico.plot(t_plot, x_plot, color='dodgerblue', 
                         label=f"Range [{tmin:.3f}-{tmax:.3f}]s", lw=0.6)
        self.grafico.plot(t_plot, x_smth_plot, color='orange', 
                         label='Smoothed', lw=0.6)
        self.grafico.legend(fontsize=6)
        self.grafico.figure.canvas.draw()
    
    def plot_ux(self):
        """
        Calculate and plot potential of mean force U(x)
        
        Computes free energy landscape from position histogram and identifies
        folded/unfolded states and transition barrier
        """
        # CHANGED: Renamed from Plot_Ux, improved error handling
        if not self.file_path:
            QMessageBox.warning(self, "Error", "No file loaded")
            return
        
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        
        pulse_num = self.Pulse_num.value()
        x = np.array(data[f"Pulse_Number_{pulse_num}"]["z"])
        
        # CHANGED: Calculate histogram and potential
        counts, bin_edges = np.histogram(x, bins=100, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        P_x = counts + EPSILON
        U_x = -np.log(P_x)
        U_x -= np.min(U_x)
        
        # CHANGED: Find minima and maxima
        min_idx, _ = find_peaks(-U_x, prominence=0.1, width=4, distance=4)
        max_idx, _ = find_peaks(U_x, prominence=0.2, width=4, distance=4)
        
        if len(min_idx) < 2 or len(max_idx) < 1:
            QMessageBox.warning(self, "Warning", 
                              "Insufficient minima/maxima found for rate calculation")
            return
        
        # CHANGED: Store data for later use
        self.counts = counts
        self.bin_centers = bin_centers
        self.U_x = U_x
        
        # CHANGED: Plot results
        plt.figure(figsize=(6, 4))
        plt.plot(bin_centers, U_x, marker='o', alpha= 0.8, color='deepskyblue', label='U(x)')
        plt.scatter(bin_centers[min_idx[0]], U_x[min_idx[0]],
                   color='lime', s=20, 
                   label=f"Folded ({bin_centers[min_idx[0]]:.2f} nm)")
        plt.scatter(bin_centers[min_idx[1]], U_x[min_idx[1]],
                   color='limegreen', s=20,
                   label=f"Unfolded ({bin_centers[min_idx[1]]:.2f} nm)")
        plt.scatter(bin_centers[max_idx[0]], U_x[max_idx[0]],
                   color='red', s=40,
                   label=f"Barrier ({bin_centers[max_idx[0]]:.2f} nm)")
        plt.xlabel("Extension x (nm)")
        plt.ylabel("Free Energy U(x) [kBT]")
        plt.title(f"Potential of Mean Force (Pulse {pulse_num})")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
    
    def dwell_analysis(self):
        """
        Analyze dwell times in folded/unfolded states
        
        Uses threshold-based state assignment to calculate folding/unfolding rates
        from mean dwell times in each state
        """
        # CHANGED: Improved error handling
        try:
            thr_down = float(self.lineEdit_thr_down.text())
            thr_up = float(self.lineEdit_thr_up.text())
        except (ValueError, AttributeError) as e:
            QMessageBox.warning(self, "Error", f"Invalid threshold values: {e}")
            return
        
        x, forces, times = self.load_json()
        if x is None:
            return
        
        pulse_num = self.Pulse_num.value()
        x_pulse = x[0]
        t = times[0]
        smoothing_window = self.smooth.value()
        x_sm = savgol_filter(x_pulse, smoothing_window, 4)
        
        # CHANGED: State assignment (0=down/folded, 1=up/unfolded)
        state = np.full_like(x_sm, np.nan)
        state[x_sm < thr_down] = 0
        state[x_sm > thr_up] = 1
        
        # CHANGED: Calculate dwell times
        dwell_up, dwell_down = [], []
        last_state, last_t = None, None
        
        for i in range(len(state)):
            if np.isnan(state[i]):
                continue
            
            if last_state is None:
                last_state = state[i]
                last_t = t[i]
                continue
            
            if state[i] != last_state:
                dwell = t[i] - last_t
                if last_state == 0:
                    dwell_down.append(dwell)
                elif last_state == 1:
                    dwell_up.append(dwell)
                last_state = state[i]
                last_t = t[i]
        
        # CHANGED: Plot histograms in GUI
        self.dwell_down_canvas.figure.clear()
        ax_down = self.dwell_down_canvas.figure.subplots()
        ax_down.hist(dwell_down, bins=30, color='dodgerblue', edgecolor='black')
        ax_down.set_xlabel("Dwell time (folded) [s]", fontsize=5)
        ax_down.set_ylabel("Count", fontsize=5)
        ax_down.set_title("Folded State", fontsize=6)
        self.dwell_down_canvas.draw()
        
        self.dwell_up_canvas.figure.clear()
        ax_up = self.dwell_up_canvas.figure.subplots()
        ax_up.hist(dwell_up, bins=30, color='orangered', edgecolor='black')
        ax_up.set_xlabel("Dwell time (unfolded) [s]", fontsize=5)
        ax_up.set_ylabel("Count", fontsize=5)
        ax_up.set_title("Unfolded State", fontsize=6)
        self.dwell_up_canvas.draw()
        
        # CHANGED: Calculate and display rates
        mean_dwell_up = np.mean(dwell_up) if len(dwell_up) > 0 else np.nan
        mean_dwell_down = np.mean(dwell_down) if len(dwell_down) > 0 else np.nan
        rate_fold = 1 / mean_dwell_up if mean_dwell_up > 0 else np.nan
        rate_unfold = 1 / mean_dwell_down if mean_dwell_down > 0 else np.nan
        
        self.label_rate_fold.setText(f"Folding rate: {rate_fold:.4g} s⁻¹")
        self.label_rate_unfold.setText(f"Unfolding rate: {rate_unfold:.4g} s⁻¹")
        
        print(f"Mean dwell unfolded: {mean_dwell_up:.4f} s → Rate fold: {rate_fold:.4g} s⁻¹")
        print(f"Mean dwell folded: {mean_dwell_down:.4f} s → Rate unfold: {rate_unfold:.4g} s⁻¹")
    
    def mfpt_analysis(self, x, U, D, prominence=PEAK_PROMINENCE, 
                     width=PEAK_WIDTH, distance=PEAK_DISTANCE):
        """
        Calculate rates using Mean First Passage Time (MFPT) from U(x)
        
        Args:
            x: Position array [nm]
            U: Potential energy array [kBT]
            D: Diffusion coefficient [nm²/s]
            prominence, width, distance: Peak finding parameters
            
        Returns:
            tuple: (rate_folding, rate_unfolding, dG_fold, dG_unfold, 
                   x_folded, x_unfolded, x_barrier)
        """
        # CHANGED: Improved interpolation handling
        mask = np.isfinite(U)
        bin_c_clean = x[mask]
        U_clean = U[mask]
        U_interp = interp1d(bin_c_clean, U_clean, kind='cubic', 
                           fill_value='extrapolate')
        
        min_idx, _ = find_peaks(-U, prominence=prominence, width=width, 
                               distance=distance)
        max_idx, _ = find_peaks(U, prominence=prominence, width=width, 
                               distance=distance)
        
        if len(min_idx) < 2 or len(max_idx) < 1:
            print("Insufficient minima/maxima for rate calculation")
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        
        # CHANGED: Extract state positions and energies
        x_folded = x[min_idx[0]]
        x_unfolded = x[min_idx[1]]
        x_barrier = x[max_idx[0]]
        
        U_fold = U[min_idx[0]]
        U_barrier = U[max_idx[0]]
        U_unfold = U[min_idx[1]]
        
        dG_fold = U_barrier - U_unfold
        dG_unfold = U_barrier - U_fold
        
        # CHANGED: MFPT calculation (folded → unfolded)
        xs = np.linspace(x_folded - 5, x_unfolded + 5, 4000)
        Us = U_interp(xs)
        expU = np.exp(Us)
        exp_minusU = np.exp(-Us)
        inner = np.cumsum(exp_minusU) * (xs[1] - xs[0])
        outer = np.sum(expU * inner) * (xs[1] - xs[0])
        mfpt_fold = outer / D
        
        # CHANGED: MFPT calculation (unfolded → folded)
        xs2 = np.linspace(x_unfolded + 5, x_folded - 5, 4000)
        Us2 = U_interp(xs2)
        expU2 = np.exp(Us2)
        exp_minusU2 = np.exp(-Us2)
        inner2 = np.cumsum(exp_minusU2) * (xs2[1] - xs2[0])
        outer2 = np.sum(expU2 * inner2) * (xs2[1] - xs2[0])
        mfpt_unfold = outer2 / D
        
        rate_folding = 1 / mfpt_unfold if mfpt_unfold > 0 else np.nan
        rate_unfolding = 1 / mfpt_fold if mfpt_fold > 0 else np.nan
        
        return (rate_folding, rate_unfolding, dG_fold, dG_unfold, 
                x_folded, x_unfolded, x_barrier)
    
    def deconvolve_ux(self):
        """
        Deconvolve U(x) using Richardson-Lucy algorithm
        
        Removes instrumental broadening from potential energy landscape
        and calculates kinetic rates using MFPT
        """
        # CHANGED: Renamed from deconvolvi_Ux, better error handling
        if self.counts is None or self.bin_centers is None:
            QMessageBox.warning(self, "Error", 
                              "No U(x) data. Please plot U(x) first.")
            return
        
        try:
            sigma = float(self.sigma_input.text())
            n_iter = int(self.niter_input.text())
        except (ValueError, AttributeError) as e:
            QMessageBox.warning(self, "Error", f"Invalid parameters: {e}")
            return
        
        # CHANGED: Build Gaussian PSF kernel
        L = int(6 * sigma + 1)
        if L % 2 == 0:
            L += 1
        x = np.linspace(-3 * sigma, 3 * sigma, L)
        kernel = np.exp(-x**2 / (2 * sigma**2))
        kernel /= kernel.sum()
        kernel = fftshift(kernel)
        
        # CHANGED: Perform Richardson-Lucy deconvolution
        from skimage.restoration import richardson_lucy
        restored = richardson_lucy(self.counts, kernel, num_iter=n_iter, clip=False)
        
        # CHANGED: Renormalize and compute deconvolved U(x)
        restored /= np.sum(restored) * (self.bin_centers[1] - self.bin_centers[0])
        U_dec = -np.log(restored + EPSILON)
        
        # CHANGED: Align minimum positions
        shift = (self.bin_centers[np.argmin(self.U_x)] - 
                self.bin_centers[np.argmin(U_dec)])
        bin_centers_shifted = self.bin_centers + shift
        U_dec_shifted = U_dec - np.min(U_dec)
        
        # CHANGED: Store for later use
        self.bin_centers_dec = bin_centers_shifted
        self.U_dec_shifted = U_dec_shifted
        
        # CHANGED: Plot comparison
        plt.figure(figsize=(6, 4))
        plt.plot(self.bin_centers, self.U_x, label="Original U(x)", 
                color="green", linewidth=2)
        plt.plot(bin_centers_shifted, U_dec_shifted, 
                label=f"Deconvolved (σ={sigma}, iter={n_iter})", 
                color="red", linewidth=2)
        plt.xlabel('Extension [nm]', fontsize=10)
        plt.ylabel('U(x) [kBT]', fontsize=10)
        plt.title('Potential Energy Profile - Deconvolution')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # CHANGED: Calculate MFPT rates
        D = self.diff_coeff.value()
        rate_folding, rate_unfolding, *_ = self.mfpt_analysis(
            self.bin_centers_dec, self.U_dec_shifted, D)
        
        self.label_mfpt_fold.setText(f"MFPT Folding rate: {rate_folding:.4g} s⁻¹")
        self.label_mfpt_unfold.setText(f"MFPT Unfolding rate: {rate_unfolding:.4g} s⁻¹")
    
    def hmm_analysis(self):
        """
        Perform Hidden Markov Model state analysis
        
        Uses Gaussian HMM to identify discrete states and calculate
        transition rates from dwell time distributions
        """
        # CHANGED: Renamed from hmm, improved documentation
        x, forces, times = self.load_json()
        if x is None:
            return
        
        pulse_num = self.Pulse_num.value()
        x_pulse = x[0]
        t = times[0]
        smoothing_window = self.smooth.value()
        x_smooth = savgol_filter(x_pulse, smoothing_window, 4)
        
        # CHANGED: Prepare data for HMM
        X = x_smooth.reshape(-1, 1)
        n_states = DEFAULT_HMM_STATES
        
        # CHANGED: Initialize with K-means clustering
        kmeans = KMeans(n_clusters=n_states, n_init=10, random_state=30)
        kmeans.fit(X)
        init_means = kmeans.cluster_centers_
        print(f"K-means initial means: {init_means.ravel()}")
        
        # CHANGED: Fit Gaussian HMM
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = hmm.GaussianHMM(
            n_components=n_states, 
            covariance_type="diag",
            n_iter=300,
            random_state=30,
            init_params=''
        )
        model.means_ = scaler.transform(init_means)
        model.fit(X_scaled)
        states = model.predict(X_scaled)
        
        # CHANGED: Order states by mean position (0=folded, 1=unfolded)
        means_real = np.array([X[states == s].mean() for s in range(n_states)])
        order = np.argsort(means_real)
        states_ordered = np.zeros_like(states)
        for new_idx, old_idx in enumerate(order):
            states_ordered[states == old_idx] = new_idx
        states = states_ordered
        
        # CHANGED: Extract dwell times
        self.dwell_times_list = []
        i = 0
        while i < len(states):
            state = states[i]
            start_idx = i
            while i < len(states) and states[i] == state:
                i += 1
            end_idx = i - 1
            start_time = float(t[start_idx])
            end_time = float(t[end_idx])
            dwell_time = end_time - start_time
            self.dwell_times_list.append((start_time, end_time, dwell_time, state))
        
        # CHANGED: Group dwell times by state
        dwell_secs = {}
        for state in range(n_states):
            times_state = [dt[2] for dt in self.dwell_times_list if dt[3] == state]
            dwell_secs[state] = np.array(times_state) if times_state else np.array([])
        
        # CHANGED: Calculate rates with error estimates
        rates = {}
        for s, arr in dwell_secs.items():
            if len(arr) == 0:
                rates[s] = (np.nan, np.nan)
                continue
            mean_tau = arr.mean()
            std_tau = arr.std(ddof=1)
            rate = 1.0 / mean_tau
            std_mean = std_tau / np.sqrt(len(arr))
            rate_err = std_mean / (mean_tau**2)
            rates[s] = (rate, rate_err)
            print(f"State {s}: k = {rate:.4g} ± {rate_err:.4g} s^-1")

        rate_unfolding = rates.get(0, (np.nan, np.nan))[0]
        rate_folding = rates.get(1, (np.nan, np.nan))[0]

        self.hmm_f_rate.setText(f"HMM Folding rate: {rate_folding:.4g} s⁻¹")
        self.hmm_u_rate.setText(f"HMM Unfolding rate: {rate_unfolding:.4g} s⁻¹")

        self.hmm_states.figure.clear()
        ax1 = self.hmm_states.figure.add_subplot(111)

        if self.hmm_points.text():
            t_limit = float(self.hmm_points.text())
            n = np.searchsorted(t, t_limit)          
        else:
            n = len(t)

        t_plot = t[:n]
        x_smooth_plot = x_smooth[:n]
        x_pulse_plot = x_pulse[:n]
        states_plot = states[:n]

        ax1.plot(t_plot, x_pulse_plot, color="gray", lw=0.5, alpha=0.3, label="Original")
        ax1.plot(t_plot, x_smooth_plot, color="black", lw=0.5, alpha=0.9, label="Smoothed")

        mask_fold = (states_plot == 0)
        mask_trans = (states_plot == 1)

        ax1.scatter(t_plot[mask_fold], x_smooth_plot[mask_fold], s=2, color="blue", alpha=0.4, label="Fold (0)")
        ax1.scatter(t_plot[mask_trans], x_smooth_plot[mask_trans], s=2, color="orange", alpha=0.5, label="Transition (1)")

        ax1.set_ylabel("Position [a.u.]", fontsize=4 )
        ax1.legend(loc="upper right", fontsize=4)
        ax1.grid(True, alpha=0.3)
        self.hmm_states.figure.tight_layout()
        self.hmm_states.draw()
        self.last_t = t
        self.last_x_smooth = x_smooth
        self.last_states = states
        self.states = states



    def save_dwell_times(self):
        
        filename, _ = QFileDialog.getSaveFileName(self, "Save dwell data", "", "CSV files (*.csv)")
        if not filename:
            return

        base, ext = os.path.splitext(filename)
        dwell_path = base + "_dwell_times.csv"
        segmented_path = base + "_segmented.csv"

        dwell_folded = [dt[2] for dt in self.dwell_times_list if dt[3] == 0]
        dwell_unfolded = [dt[2] for dt in self.dwell_times_list if dt[3] == 1]

        max_len = max(len(dwell_folded), len(dwell_unfolded))  # so that lists have same length 
        dwell_folded += ["" for _ in range(max_len - len(dwell_folded))]
        dwell_unfolded += ["" for _ in range(max_len - len(dwell_unfolded))]

        with open(dwell_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Dwell_folded [s]", "Dwell_unfolded [s]"])
            for f_dw, u_dw in zip(dwell_folded, dwell_unfolded):
                writer.writerow([f"{f_dw:.6f}" if f_dw != "" else "", f"{u_dw:.6f}" if u_dw != "" else ""])

        x, forces, times = self.load_json()
        t = np.array(times[0])
        x_pulse = np.array(x[0])

        smoothing_window = self.smooth.value()
        x_smooth = savgol_filter(x_pulse, smoothing_window, 4)

        if hasattr(self, "states"):
            states = np.array(self.states)
        else:
            QMessageBox.warning(self, "Warning", "No state sequence found. Run HMM first.")
            return

        segmented_filename = filename.replace(".csv", "_segmented.csv")
        with open(segmented_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time [s]", "Extension_folded [nm]", "Extension_unfolded [nm]"])
            unique_states = np.unique(states)
            print(f"DEBUG SALVATAGGIO: Stati presenti nell'array: {unique_states}")

            for ti, xi, si in zip(t, x_smooth, states):
                current_state = int(si)
                if current_state == 0:  # folded
                    writer.writerow([f"{ti:.6f}", f"{xi:.6f}", ""])
                elif current_state == 1:  # unfolded
                    writer.writerow([f"{ti:.6f}", "", f"{xi:.6f}"])

        QMessageBox.information(
            self,
            "Saved",
            f"I file CSV sono stati salvati correttamente:\n\n"
            f"• {os.path.basename(dwell_path)}\n"
            f"• {os.path.basename(segmented_path)}"
        )




if __name__ == "__main__":
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    plt.style.use('dark_background')

    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())   
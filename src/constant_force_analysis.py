#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:16:42 2025

@author: w2040021
"""

import numpy as np
import sys
import json
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from numpy.fft import fftshift


import csv
plt.style.use('dark_background')
from IPython.display import display,HTML
display(HTML("<style>.container{width:95% !important;}</style>"))

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, QSlider, QPushButton, QLCDNumber, QVBoxLayout
from PyQt5.QtCore import QSize
from PyQt5.uic import loadUiType
from PyQt5.QtGui import QIcon

from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from scipy.signal import savgol_filter





qtCreatorFile = "UI_Jarzynski_2 - Copy.ui" # Enter .ui file here.

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
        self.setWindowTitle('Pulse ANALYZER')
        
    #    self.Pulse_slider.setValue(700)
       # self.LCD1.display(700)
        #self.Pulse_slider.valueChanged.connect(self.LCD1.display)
        #self.Pulse_num.setText('0') #MOhm 
       # self.Cm.setText('10') #pFarads
      #  self.gK.setText('8e-7') #S
     #   self.gNa.setText('1e-6') #S
      #  self.Vrest.setText('-77') #mV
        self.Open_JSON.clicked.connect(self.File_open)
            
        self.Plot_button.clicked.connect(self.Plotter)

        grafico=FigureCanvas(Figure(figsize=(8,6),constrained_layout=True, edgecolor='black', frameon=True))
        self.Plotting_box.addWidget(grafico)
        #self.addToolBar(NavigationToolbar(grafico, self))
        self.grafico=grafico.figure.subplots()
        self.grafico.set_xlabel('Time (s)', size=8)
        self.grafico.set_ylabel('Extension (nm)', size=8)
        
        x=np.linspace(0,0.03, 100)
        self.grafico.plot(x, 0*x)
        
        # grafico2=FigureCanvas(Figure(figsize=(8,6),constrained_layout=True, edgecolor='black', frameon=True))
        # self.Plotting_box_2.addWidget(grafico2)
        # #self.addToolBar(NavigationToolbar(grafico, self))
        # self.grafico2=grafico2.figure.subplots()
        # self.grafico2.set_xlabel('Time (s)', size=8)
        # self.grafico2.set_ylabel('Force (pN)', size=8)
    
        
        x=np.linspace(0,0.03, 100)
        # self.grafico2.plot(x, 0*x)

        # gui per U(x)
        self.Plot_Ux_button.clicked.connect(self.Plot_Ux)

        # dwell time rate
        self.dwell_down_canvas = FigureCanvas(Figure())
        self.dwell_up_canvas = FigureCanvas(Figure())

        lay_dwell_down = QVBoxLayout(self.widget_dwell_down)
        lay_dwell_down.setContentsMargins(0,0,0,0)
        lay_dwell_down.addWidget(self.dwell_down_canvas)

        lay_dwell_up = QVBoxLayout(self.widget_dwell_up)
        lay_dwell_up.setContentsMargins(0,0,0,0)
        lay_dwell_up.addWidget(self.dwell_up_canvas)
        self.btn_dwell.clicked.connect(self.dwell_analysis)

        # deconvolution 
        self.deconvolution_button.clicked.connect(self.deconvolvi_Ux)

        # zoom in plot
        self.btn_update.clicked.connect(self.plot_subportion)


  
    def File_open(self):
        name = QtWidgets.QFileDialog.getOpenFileName(self, "Open JSON File")
        global file_path
        file_path = name[0]
        
        return file_path
      
   
    # upload file with ramp-test data
    def Load_JSON(self):        
        with open(file_path , 'r') as f:
           d=json.load(f) 
         
      #  d = self.File_open()  
        Pulse_num=self.Pulse_num.value()
        xs, forces, times = [], [], []
        for i in range (len(d)):
            xs.append(np.array(d["Pulse_Number_"+str(Pulse_num)]["z"]))
            forces.append(np.array(d["Pulse_Number_"+str(Pulse_num)]["force"]))
            times.append(np.array(d["Pulse_Number_"+str(Pulse_num)]["time"])-d["Pulse_Number_"+str(Pulse_num)]["time"][0])    
        
        return xs, forces, times 
        
    # plot F(t) vs x(t)
    def Plotter(self):
        
        xs, forces, times=self.Load_JSON()
        Pulse_num=self.Pulse_num.value()
        xs_smth = []
        xs_smth= savgol_filter(xs, 51, 4)
        i=Pulse_num

        self.t = times[i]
        self.x = xs[i]
        self.x_smth = xs_smth[i]

        # plot in gui 
        self.grafico.clear()
        self.grafico.set_xlabel('Time (s)', size=8)
        self.grafico.set_ylabel('Extension (nm)', size=8)
           # self.grafico.axes.set_ylim([-100,50])
        self.grafico.plot(times[i], xs[i])
        self.grafico.plot(times[i], xs_smth[i])
        self.grafico.figure.canvas.draw()

        #plot out 
        # plt.figure(figsize=(13,8))
        # plt.plot(times[i], xs[i])
        # plt.plot(times[i], xs_smth[i])
        # plt.show()
           
        # self.grafico2.clear()
        # self.grafico2.set_xlabel('Time (s)', size=8)
        # self.grafico2.set_ylabel('Force (pN)', size=8)
        # self.grafico2.plot(times[i], forces[i], 'r')
        # self.grafico2.figure.canvas.draw()
        
       # self.Output.setText("hola")
        return Pulse_num, xs_smth, forces, times

    def plot_subportion(self):
        if not hasattr(self, 't') or not hasattr(self, 'x'):
            return
        try:
            tmin = float(self.edit_tmin.text())
        except:
            tmin = np.min(self.t)
        try:
            tmax = float(self.edit_tmax.text())
        except:
            tmax = np.max(self.t)
        mask = (self.t >= tmin) & (self.t <= tmax)
        t_plot = self.t[mask]
        x_plot = self.x[mask]
        x_smth_plot = self.x_smth[mask]

        # --- Plot solo sub-porzione ---
        self.grafico.clear()
        self.grafico.set_xlabel('Time (s)', size=8)
        self.grafico.set_ylabel('Extension (nm)', size=8)
        self.grafico.plot(t_plot, x_plot, color='dodgerblue', label=f"Sub-portion [{tmin}-{tmax}]")
        self.grafico.plot(t_plot, x_smth_plot, color='orange', label='Smoothed')
        self.grafico.legend()
        self.grafico.figure.canvas.draw()

        
        
    def Save_to_file(self):
        #F= self.Step_detect()
        np.savetxt("Unfolding_forces.csv", Unfolding_forces, delimiter=",")
    

    # plotting U(x) per each pulse
    def Plot_Ux(self):
        # Carica dati del pulse selezionato
        with open(file_path, 'r') as f:
            d = json.load(f)
        Pulse_num = self.Pulse_num.value()
        x = np.array(d[f"Pulse_Number_{Pulse_num}"]["z"])  # extension

        counts, bin_edges = np.histogram(x, bins=100, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        P_x = counts + 1e-12
        U_x = - np.log(P_x)
        U_x -= np.min(U_x)  # opzionale: minimo a zero
        
        # Trova minimo e massimo
        
        mask = np.isfinite(U_x)
        bin_c_clean = bin_centers[mask]
        U_clean = U_x[mask]
        U_interp = interp1d(bin_c_clean, U_clean, kind='cubic', fill_value='extrapolate')

        min_idx, _ = find_peaks(-U_x, prominence=0.1, width=6, distance=6)
        max_idx, _ = find_peaks(U_x, prominence=0.4, width=6, distance=6)

        if len(min_idx) < 2 or len(max_idx) < 1:
            print("Non trovati sufficienti minimi/massimi per calcolo rate")
            return (np.nan, np.nan, np.nan, np.nan)

        x_folded = x[min_idx[0]]
        x_unfolded = x[min_idx[1]]
        x_barrier = x[max_idx[0]]

        U_fold = U_x[min_idx[0]]
        U_barrier = U_x[max_idx[0]]
        U_unfold = U_x[min_idx[1]]

        dG_fold = U_barrier - U_unfold
        dG_unfold = U_barrier - U_fold

        # Salva risultati per uso futuro
        self.counts = counts
        self.bin_centers = bin_centers
        self.U_x = U_x
        self.Ux_minima = [
            (float(bin_centers[min_idx[0]]), float(U_x[min_idx[0]])), 
            (float(bin_centers[min_idx[1]]), float(U_x[min_idx[1]]))]
        self.Ux_maximum = (float(bin_centers[max_idx]), float(U_x[max_idx]))

        plt.figure(figsize=(6,4))
        plt.plot(bin_centers, U_x, marker='o', color='deepskyblue', label='U(x)')
        plt.scatter(bin_centers[min_idx[0]], U_x[min_idx[0]],
                    color='lime', s=80,
                    label=f"Min 1 ({bin_centers[min_idx[0]]:.2f}, {U_x[min_idx[0]]:.2f})")
        plt.scatter(bin_centers[min_idx[1]], U_x[min_idx[1]],
                    color='limegreen', s=80,
                    label=f"Min 2 ({bin_centers[min_idx[1]]:.2f}, {U_x[min_idx[1]]:.2f})")
        plt.scatter(self.Ux_maximum[0], self.Ux_maximum[1],
                    color='red', s=100,
                    label=f"Max ({self.Ux_maximum[0]:.2f}, {self.Ux_maximum[1]:.2f})")
        plt.xlabel("Extension x (nm)")
        plt.ylabel("Free Energy U(x) [pN·nm]")
        plt.title(f"Potential of Mean Force (Pulse {self.Pulse_num.value()})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def dwell_analysis(self):
        ## computing rate with dwell time -   def dwell_analysis(self):
        try:
            thr_down = float(self.lineEdit_thr_down.text())
            thr_up = float(self.lineEdit_thr_up.text())

        except Exception as e:
            QMessageBox.warning(self, "Errore soglie", f"Errore nelle soglie: {e}")
            return


        x, forces, times=self.Load_JSON()
        Pulse_num=self.Pulse_num.value()  # selezionato nella gui
        x_pulse = x[Pulse_num]
        t = times[Pulse_num]
        x_sm = savgol_filter(x_pulse, 51, 4)

        # Stati: 0 (down), 1 (up), np.nan = esclusi
        state = np.full_like(x_sm, np.nan)
        state[x_sm < thr_down] = 0
        state[x_sm > thr_up] = 1

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

        # Istogrammi (popup matplotlib)
       # DOWN
        self.dwell_down_canvas.figure.clear()
        ax_down = self.dwell_down_canvas.figure.subplots()
        ax_down.hist(dwell_down, bins=30, color='dodgerblue')
        ax_down.set_xlabel("Dwell time DOWN [s]")
        ax_down.set_ylabel("Count")
        ax_down.set_title("DOWN state")
        self.dwell_down_canvas.draw()

        # UP
        self.dwell_up_canvas.figure.clear()
        ax_up = self.dwell_up_canvas.figure.subplots()
        ax_up.hist(dwell_up, bins=30, color='orangered')
        ax_up.set_xlabel("Dwell time UP [s]")
        ax_up.set_ylabel("Count")
        ax_up.set_title("UP state")
        self.dwell_up_canvas.draw()

        # Calcola rate folding/unfolding
        mean_dwell_up = np.mean(dwell_up) if len(dwell_up) > 0 else np.nan
        mean_dwell_down = np.mean(dwell_down) if len(dwell_down) > 0 else np.nan
        rate_fold = 1 / mean_dwell_up if mean_dwell_up > 0 else np.nan
        rate_unfold = 1 / mean_dwell_down if mean_dwell_down > 0 else np.nan

        #aggionro label GUI
        self.label_rate_fold.setText(f"Dwell/Folding rate: {rate_fold:.4g} s⁻¹")
        self.label_rate_unfold.setText(f"Dwell/Unfolding rate: {rate_unfold:.4g} s⁻¹")

        # Stampa su console e aggiorna label GUI se esiste
        print(f"Mean dwell UP: {mean_dwell_up:.4f} s,  Rate fold: {rate_fold:.4g} 1/s")
        print(f"Mean dwell DOWN: {mean_dwell_down:.4f} s,  Rate unfold: {rate_unfold:.4g} 1/s")
        if hasattr(self, "label_rate_fold"):
            self.label_rate_fold.setText(f"Rate fold: {rate_fold:.4g} s⁻¹")
        if hasattr(self, "label_rate_unfold"):
            self.label_rate_unfold.setText(f"Rate unfold: {rate_unfold:.4g} s⁻¹")

# Deconvlution 
    def jansson_deconvolution(self, observed, kernel, iterations=700, alpha=0.02):
            from scipy.signal import fftconvolve
            estimate = np.copy(observed).astype(np.float64)
            for _ in range(iterations):
                reconvolved = fftconvolve(estimate, kernel, mode='same')
                correction = observed - reconvolved
                update = alpha * correction * estimate
                estimate += update
                estimate[estimate < 0] = 0
            return estimate


    # rate mfpt on deconvlved Ux
    def mfpt_analysis(self, x, U, D, prominence=0.3, width=6, distance=6, debug=True):

        mask = np.isfinite(U)
        bin_c_clean = x[mask]
        U_clean = U[mask]
        U_interp = interp1d(bin_c_clean, U_clean, kind='cubic', fill_value='extrapolate')

        min_idx, _ = find_peaks(-U, prominence=prominence, width=width, distance=distance)
        max_idx, _ = find_peaks(U, prominence=prominence, width=width, distance=distance)
        if len(min_idx) < 2 or len(max_idx) < 1:
            print("Non trovati sufficienti minimi/massimi per calcolo rate")
            return (np.nan, np.nan, np.nan, np.nan)

        # Trova le posizioni chiave
        x_folded = x[min_idx[0]]
        x_unfolded = x[min_idx[1]]
        x_barrier = x[max_idx[0]]

        U_fold = U[min_idx[0]]
        U_barrier = U[max_idx[0]]
        U_unfold = U[min_idx[1]]

        dG_fold = U_barrier - U_unfold
        dG_unfold = U_barrier - U_fold

        # Folding: da folded → unfolded (attraversando la barriera)
        xs = np.linspace(x_folded - 5, x_unfolded +5, 4000)
        Us = U_interp(xs)
        expU = np.exp(Us)
        exp_minusU = np.exp(-Us)
        inner = np.cumsum(exp_minusU) * (xs[1] - xs[0])
        outer = np.sum(expU * inner) * (xs[1] - xs[0])
        mfpt_fold = outer / D

        # Unfolding: da unfolded → folded (direzione opposta)
        xs2 = np.linspace(x_unfolded+5, x_folded-5, 4000)
        Us2 = U_interp(xs2)
        expU2 = np.exp(Us2)
        exp_minusU2 = np.exp(-Us2)
        inner2 = np.cumsum(exp_minusU2) * (xs2[1] - xs2[0])
        outer2 = np.sum(expU2 * inner2) * (xs2[1] - xs2[0])
        mfpt_unfold = outer2 / D

        # I rate sono l'inverso degli MFPT nei due sensi
        rate_folding = 1 / mfpt_unfold if mfpt_unfold > 0 else np.nan
        rate_unfolding = 1 / mfpt_fold if mfpt_fold > 0 else np.nan
        
        return rate_folding, rate_unfolding, dG_fold, dG_unfold, x_folded, x_unfolded, x_barrier

    def deconvolvi_Ux(self):
        try:
            sigma = float(self.sigma_input.text())
            # alpha = float(self.alpha_input.text())
            n_iter = int(self.niter_input.text())
        except Exception as e:
            print(f"Errore nei parametri: {e}")
            return

        # Costruisci la PSF gaussiana (kernel)
        L = int(6 * sigma + 1)
        if L % 2 == 0:
            L += 1
        x = np.linspace(-3*sigma, 3*sigma, L)
        kernel = np.exp(-x**2 / (2*sigma**2))
        kernel /= kernel.sum()
        kernel = fftshift(kernel)

        # Usa i dati salvati nell'oggetto
        # try con RL
        from skimage.restoration import richardson_lucy
        restored= richardson_lucy(self.counts, kernel, num_iter=n_iter, clip=False)
        # restored = self.jansson_deconvolution(self.counts, kernel, iterations=n_iter, alpha=alpha)
        # Rinormalizza
        bin_centers = self.bin_centers
        restored /= np.sum(restored) * (bin_centers[1] - bin_centers[0])
        U_dec = -np.log(restored + 1e-10)
        shift = bin_centers[np.argmin(self.U_x)] - bin_centers[np.argmin(U_dec)]
        bin_centers_shifted = bin_centers + shift
        U_dec_shifted = U_dec - np.min(U_dec)

        self.bin_centers_dec = bin_centers_shifted
        self.U_dec_shifted = U_dec_shifted

        # Plot 
        plt.figure(figsize=(6, 4))
        plt.plot(bin_centers, self.U_x, label=" (U(x))", color="green")
        plt.plot(bin_centers_shifted, U_dec_shifted, label=f"deconvolution", color="red")
        plt.xlabel('x [nm]')
        plt.ylabel('U(x) [kBT]')
        # plt.title(f'Potential energy profile\nDeconvolution: alpha={alpha}, n_iter={n_iter}')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # rate mfpt
        rate_folding, rate_unfolding, *_ = self.mfpt_analysis(
            self.bin_centers_dec,
            self.U_dec_shifted,
            D=3000)       
        self.label_mfpt_fold.setText(f"MFPT Folding rate: {rate_folding:.4g} s⁻¹")
        self.label_mfpt_unfold.setText(f"MFPT Unfolding rate: {rate_unfolding:.4g} s⁻¹")

        if hasattr(self, "label_mfpt_fold"):
            self.label_mfpt_fold.setText(f"MFPT Folding rate: {rate_folding:.4g} s⁻¹")
        if hasattr(self, "label_mfpt_unfold"):
            self.label_mfpt_unfold.setText(f"MFPT Unfolding rate: {rate_unfolding:.4g} s⁻¹")


if __name__ == "__main__":
     app = QtWidgets.QApplication(sys.argv)
     window = MyApp()
     window.show()
     sys.exit(app.exec_())   

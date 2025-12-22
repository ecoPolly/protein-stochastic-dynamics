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
from scipy.stats import norm
from scipy.optimize import curve_fit


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


qtCreatorFile = "UI_Jarzynski_2_ramp.ui" # Enter .ui file here.

Ui_MainWindow, QtBaseClass = loadUiType(qtCreatorFile)        

Unfolding_forces = []

global file_path
file_path = ""

# UNFOLDING FORCES DISTRIBUTIONS
def bell_model(F, x_beta, k0, r):
    kBT = 4.114  # pN·nm
    F = np.asarray(F)

    exponent = (F * x_beta) / kBT
    exp_term = np.exp(exponent)
    suppression = np.exp(- ((k0 * kBT) / (r * x_beta)) * (exp_term - 1))
    return (k0 / r) * exp_term * suppression



def fit_bell_evans_model(forces,r):
    hist, bin_edges = np.histogram(forces, bins=18, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    popt, _ = curve_fit(lambda F, x, k0: bell_model(F, x, k0, r), 
                           bin_centers, 
                           hist)
    x_beta, k0 = popt
    y_fit = bell_model(bin_centers, x_beta, k0, r)
   
    print(f"[DEBUG] y_fit: {y_fit[:5]} ... max={np.max(y_fit)}, min={np.min(y_fit)}")
    return x_beta, k0, bin_centers, hist, y_fit

def detect_jump(xs_smth, threshold, drift=0.02):
    s_pos = np.zeros(len(xs_smth))
    s_neg = np.zeros(len(xs_smth))
    for i in range(20, len(xs_smth)):
        diff = xs_smth[i] - xs_smth[i - 1]
        s_pos[i] = max(0, s_pos[i - 1] + diff - drift)
        s_neg[i] = min(0, s_neg[i - 1] + diff + drift)
        if s_pos[i] > threshold or abs(s_neg[i]) > threshold:
            return i  # ritorna primo salto
    return None  


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        super().__init__()
        
        self.setupUi(self)
        self.setWindowTitle('RAMP ANALYZER')

        self.Open_JSON.clicked.connect(self.File_open)
            
        self.Plot_button.clicked.connect(self.Plotter)
        self.Step.clicked.connect(self.Step_detect)
        self.Histo.clicked.connect(self.Do_histogram)
        self.Save_file.clicked.connect(self.Save_to_file)
        # self.borrar.clicked.connect(self.remove_value)
       
        self.Output.setText("ramp")

        grafico=FigureCanvas(Figure(figsize=(8,6),constrained_layout=True, edgecolor='black', frameon=True))
        self.Plotting_box.addWidget(grafico)
        self.grafico=grafico.figure.subplots()
        self.grafico.set_xlabel('Time (s)', size=8)
        self.grafico.set_ylabel('Extension (nm)', size=8)
        
        x=np.linspace(0,0.03, 100)
        self.grafico.plot(x, 0*x)
        
        grafico3=FigureCanvas(Figure(figsize=(8,6),constrained_layout=True, edgecolor='black', frameon=True))
        self.Plotting_box_3.addWidget(grafico3)
        #self.addToolBar(NavigationToolbar(grafico, self))
        self.grafico3=grafico3.figure.subplots()
        self.grafico3.set_xlabel('Unfolding Force (pN)', size=8)
        self.grafico3.set_ylabel('Counts', size=8)
        
        x=np.linspace(0,0.03, 100)
        # self.grafico2.plot(x, 0*x)
        self.r_box = self.findChild(QtWidgets.QSpinBox, "r_box")

  
    def File_open(self):
        name = QtWidgets.QFileDialog.getOpenFileName(self, "Open JSON File")
        global file_path
        file_path = name[0]
      
        return file_path
      
    # upload file with ramp-test data
    def Load_JSON(self):        
        with open(file_path , 'r') as f:
           self.data=json.load(f) 
 
        Pulse_num=self.Pulse_num.value()
        xs, forces, times = [], [], []
        for i in range (len(self.data)):
            xs.append(np.array(self.data["Pulse_Number_"+str(Pulse_num)]["z"]))
            forces.append(np.array(self.data["Pulse_Number_"+str(Pulse_num)]["force"]))
            times.append(np.array(self.data["Pulse_Number_"+str(Pulse_num)]["time"])- self.data["Pulse_Number_"+str(Pulse_num)]["time"][0])    
        
        return xs, forces, times 
        
    # plot F(t) vs x(t)
    def Plotter(self):
        
        xs, forces, times=self.Load_JSON()
        Pulse_num=self.Pulse_num.value()
        xs_smth = []
        xs_smth = [savgol_filter(x, 51, 4) for x in xs]
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
        
        return Pulse_num, xs_smth, forces, times

    def Step_detect(self):
        Pulse_num, xs_smth, forces, times = self.Plotter()
        threshold = self.Thr.value()
        jump_i = detect_jump(xs_smth[Pulse_num], threshold)

        if jump_i is not None:
            self.grafico.axvline(x=times[Pulse_num][jump_i], lw=1)
            self.grafico.figure.canvas.draw()
            F = round(forces[Pulse_num][jump_i], 2)
            self.Output.setText("Unfolding force=" + str(F) + " pN")
            Unfolding_forces.append(forces[Pulse_num][jump_i])

        return Unfolding_forces

    # Unf_F hist 
    def Do_histogram(self):
        global Unfolding_forces
        Unfolding_forces = []

        with open(file_path , 'r') as f:
            data = json.load(f)

        threshold = self.Thr.value()

        for key in data:
            try:
                xs = np.array(data[key]["z"])
                forces = np.array(data[key]["force"])
            except KeyError:
                continue
            xs_smth = savgol_filter(xs, 51, 4)
            jump_i = detect_jump(xs_smth, threshold)
            if jump_i is not None:
                Unfolding_forces.append(forces[jump_i])

        self.grafico3.clear()
        self.grafico3.set_xlabel('Unfolding Force (pN)', size=8)
        self.grafico3.set_ylabel('Counts', size=8)

        counts, bins, _ = self.grafico3.hist(Unfolding_forces, bins=18, color='blue', alpha=0.5, edgecolor='black', density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # fit plot 
        r = self.r_box.value()
        print("Pulling rate (r):", r)
        x_beta, k0, bin_centers, counts, y_fit = fit_bell_evans_model(Unfolding_forces, r)        
        y_fit *= np.max(counts)/ np.max(y_fit)
        self.grafico3.plot(bin_centers, y_fit, color='red', lw=2, label='Bell-Evans fit')        

        # mu, std = norm.fit(Unfolding_forces)
        # gauss_fit = norm.pdf(bin_centers, mu, std)
        # self.grafico3.plot(bin_centers, gauss_fit, color='orange', lw=2, label='Gaussian fit')

        self.grafico3.legend()
        self.grafico3.figure.canvas.draw()

        self.xdag_label.setText(f"x_dag estimate: {x_beta:.2f} nm")
        self.k0_label.setText(f"k0 estimated: {k0:.2f} 1/s")
        self.Output.setText(f"Ramps analyzed: {len(data)}, Events: {len(Unfolding_forces)}")

    def Save_to_file(self):
        #F= self.Step_detect()
        np.savetxt("Unfolding_forces.csv", Unfolding_forces, delimiter=",")
        
    def remove_value(self):
        del Unfolding_forces[-1]


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())   

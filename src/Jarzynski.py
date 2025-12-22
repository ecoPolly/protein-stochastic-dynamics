#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

import sys
import json
import os
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
import matplotlib.pyplot as plt
import csv
plt.style.use('dark_background')
from IPython.display import display,HTML
display(HTML("<style>.container{width:95% !important;}</style>"))
import scipy


from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, QSlider, QPushButton, QLCDNumber
from PyQt5.QtCore import QSize
from PyQt5.uic import loadUiType
from PyQt5.QtGui import QIcon

from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit





qtCreatorFile = "UI_Jarzynski_dG_W.ui" # Enter file here.

Ui_MainWindow, QtBaseClass = loadUiType(qtCreatorFile)        

Total_Work = []
Work_polymer = []

global file_path
file_path = ""

def WLC(x, Lc, lp, L0):
    return (4.11 / lp) * (0.25 * (1 - (x - L0) / Lc) ** (-2) + (x - L0) / Lc - 0.25)


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    


    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        super().__init__()
        
        self.setupUi(self)
        
        self.Open_JSON.clicked.connect(self.File_open)
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
            
        ## SELF BUTTONS 
        self.Plot_button.clicked.connect(self.Plotter)
        self.Plot_avg_traj.clicked.connect(self.Integrate_curve)
        self.wlc_work.clicked.connect(self.WLC_work)
        #self.Histo.clicked.connect(self.Do_histogram)
        self.Save_file.clicked.connect(self.Save_to_file)
   
        grafico=FigureCanvas(Figure(figsize=(8,6),constrained_layout=True, edgecolor='black', frameon=True))
        self.Plotting_box.addWidget(grafico)
        self.grafico=grafico.figure.subplots()
        self.grafico.set_xlabel('Extension (nm)', size=8)
        self.grafico.set_ylabel('Force (pN)', size=8)
        
        x=np.linspace(0,0.03, 100)
        self.grafico.plot(x, 0*x)

        # initial guess for gui 
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
        

        self.FJC_W.clicked.connect(self.FJC)
        self.compute_sgW.clicked.connect(self.sg_Wtot)

        grafico_hist = FigureCanvas(Figure(figsize=(8, 6), constrained_layout=True))
        self.W_hist.layout().addWidget(grafico_hist)  
        self.grafico_hist = grafico_hist.figure.subplots()

        # global fit 
        self.global_fit.clicked.connect(self.WLC_global_fit)    
        # clean work
        self.Wp_strech.clicked.connect(self.W_p_stretch)   
        self.Wh_strech.clicked.connect(self.W_h_stretch)
        self.clean_w.clicked.connect(self.clean_W)
        self.dG_distribution.clicked.connect(self.dG)

        # VARIABILI DI ISTANZA 
        self.d = None 

  
    def File_open(self):
        name = QtWidgets.QFileDialog.getOpenFileName(self, "Open JSON File")
        global file_path
        file_path = name[0]
        return file_path
      

    def Load_JSON(self):        
        with open(file_path , 'r') as f:
           self.d=json.load(f) 
      #  d = self.File_open()  
        xs, forces, times, current = [], [], [], []
        # Pulse_num=self.Pulse_num.value()
        ## == SHIFT ON X FOR 1st 300 pulses ==
        for i in range (0,len(self.d)-1):    # METTI , 2 PER ALTOR FILE 
            xs_array = (np.array(self.d["Pulse_Number_"+str(i)]["z"]))
            forces_array= (np.array(self.d["Pulse_Number_"+str(i)]["force"]))
            times_array=(np.array(self.d["Pulse_Number_"+str(i)]["time"])- self.d["Pulse_Number_"+str(i)]["time"][0])    
            current_array= (np.array(self.d["Pulse_Number_"+str(i)]["current"]))
            I = current_array
            # == SHIFT ON X SPECIFIC FOR file: 20250411_R3_5pNs_ramps_1.Json ==
            # if i < 300:
            #     xs_array += 10
            # if i > 300 and i <600:
            #     xs_array += 5
            F = (1.504e-5)* I**2 - 0.0133*I
            xs.append(xs_array)
            forces.append(F)
            times.append(times_array)
            current.append(current_array)

        # plt.figure (13)
        # for i in range (500, 600,2):
        #     plt.plot(xs[i], forces[i],alpha=0.5)
        # plt.show()
        # plt.figure (14)
        # for i in range (600, 700,2):
        #     plt.plot(xs[i], forces[i],alpha=0.5)
        # plt.show()
        # plt.figure (15)
        # for i in range (700, 800,2):
        #     plt.plot(xs[i], forces[i],alpha=0.5)
        # plt.show()
        # plt.figure (16)
        # for i in range (800, 900,2):
        #     plt.plot(xs[i], forces[i],alpha=0.5)
        # plt.show()

        print (len(xs), len(forces), len(times))
        return xs, forces, times, current
        
        
    def Plotter(self, plot_avg=False):
        xs, forces, times, current = self.Load_JSON()  # even idx data already extracted 
        Pulse_num = self.Pulse_num.value()
        # xs_smth = savgol_filter(xs, 51, 4)
        xs_smth = [savgol_filter(arr, 51 , 4) for arr in xs]

        # for i in range(len(forces)):
        #     forces[i] = np.array((3.48e-5) * current[i]**2 - 0.0029 * current[i])

        self.grafico.clear()
        self.grafico.set_xlabel('Extension (nm)', size=6)
        self.grafico.set_ylabel('Force (pN)', size=6)
        # self.grafico.set_xlim(0, 105)

        if plot_avg:
            x_avg, f_avg = self.avg_traj(xs, forces, n= len(forces) )
            self.grafico.plot(x_avg, f_avg, '-', lw=1, alpha=0.5,  color='orange')
        else:
            self.grafico.plot(xs[Pulse_num], forces[Pulse_num], alpha=0.5)
            self.grafico.plot(xs_smth[Pulse_num], forces[Pulse_num], alpha=0.5)
        self.grafico.figure.canvas.draw()

        return Pulse_num, xs_smth, forces, times

    
    
    def avg_traj(self, xs, forces, n):
            print ("n", n)
            # for i in range(n):
            #     print(f"len xs[{i}]:", len(xs[i]))
            min_len = min(len(xs[i]) for i in range(n))  # prendo stesso numero di punti per pulse 
            # print ("min len xs:", min_len)
            xs_array = np.array([xs[i][:min_len] for i in range(n)])
            fs_array = np.array([forces[i][:min_len] for i in range(n)])
            x_avg = np.mean(xs_array, axis=0)
            f_avg = np.mean(fs_array, axis=0)

            return x_avg, f_avg
    
    # load for each sg work tot - USALA QUANDO DEVI CALIBRARE FORZA 
    def Load(self):
        with open(file_path, 'r') as f:
            d = json.load(f)

        xs, forces_calc, times, current = [], [], [], []

        for i in range(0,len(d)):  # ,2 
            key = f"Pulse_Number_{i}"
            x = np.array(d[key]["z"])
            t = np.array(d[key]["time"]) - d[key]["time"][0]
            I = np.array(d[key]["current"])
            F = np.array(d[key]["force"])

            # calcolo forza teorica per ogni pulse
            # F = (3.48e-5) * I**2 - 0.0029 * I
            F = (1.504e-5)* I**2 - 0.0133*I
            xs.append(x)
            times.append(t)
            current.append(I)
            forces_calc.append(F)
        print ("Load check in sg W ", len(xs), len(forces_calc), len(times))


        return xs, forces_calc, times, current  # indici già pari sistemati 


    def sg_Wtot(self):
        xs, forces, times, current = self.Load_JSON()
        work_list = []

        xs_smth = [savgol_filter(x, 51, 4) for x in xs]

        for idx in range(len(forces)):
            xmin = next(i for i in range(1, len(forces[idx]))
                        if forces[idx][i] > self.lower_limit.value() and forces[idx][i - 1] < self.lower_limit.value())
            xmax = next(i for i in range(1, len(forces[idx]))
                        if forces[idx][i] > self.upper_limit.value() and forces[idx][i - 1] < self.upper_limit.value())

            # print(f"Pulse {idx}: xmin={xmin}, xmax={xmax}, force range=({forces[idx][xmin]}, {forces[idx][xmax]})")
            integrate_force = forces[idx][xmin:xmax]  # idx = pulse ; [:] -> index 
            integrate_x = xs [idx][xmin:xmax]
            # W = scipy.integrate.simpson(integrate_force, integrate_x)

            W = 0.0
            for i in range(len(integrate_x)-1): 
                dx = integrate_x[i+1] - integrate_x[i]
                avg_height = (np.abs(integrate_force[i]) + np.abs(integrate_force[i+1])) / 2.0   # np.abs 
                W += dx * avg_height

            work_list.append(W)


        counts, bin_edges = np.histogram(work_list, bins=700)
        # for i in range(len(counts)):          
        #     print(f"Bin {i+1} (range {bin_edges[i]} to {bin_edges[i+1]}): {counts[i]} frequencies")

        self.grafico_hist.set_xlabel('Total Work  [pNnm]', size=4)
        self.grafico_hist.hist(work_list, bins=200, edgecolor='black', alpha=0.6, color='orange')
        # self.grafico_hist.set_xlim(220, 700)
        # self.grafico_hist.set_ylim(0, 20)

        self.grafico_hist.figure.canvas.draw()

        plt.figure(10)
        x_range = np.linspace (0,len(forces), len(forces))
        plt.plot(x_range, work_list, lw=0.5)
        plt.show()

        print ("mean W:", np.mean(work_list), "median:", np.median(work_list))

        return work_list



    def Integrate_curve(self):
        Pulse_num, xs, forces, times = self.Plotter(plot_avg=True)  # togli contenuto quando vuoi plot sg 
        x_avg, f_avg = self.avg_traj(xs, forces, n=len(forces))
    
        # == calcolo limits su traj media == 
        for i in range(1, len(f_avg)):
            if f_avg[i] > self.lower_limit.value() and f_avg[i - 1] < self.lower_limit.value():
                xmin = i
            if f_avg[i] > self.upper_limit.value() and f_avg[i - 1] < self.upper_limit.value():
                xmax = i

        xmin_reale = x_avg[xmin]
        xmax_reale = x_avg[xmax]
        print("xmin reale:", xmin_reale, "xmax reale:", xmax_reale)

        return xmin_reale, xmax_reale, x_avg, f_avg
    

    def compute_WLC_dx(self, xmin_reale, xmax_reale, Lc_dx, Lp_dx, L0):
            x_interp = np.linspace(xmin_reale, xmax_reale, 1000)  
            x_vals = []
            y_vals = []

            for x in x_interp:
                try:
                    F = WLC(x, Lc_dx, Lp_dx, L0)
                    x_vals.append(x)
                    y_vals.append(F)
                except:
                    continue
            
            area_wlc = scipy.integrate.simpson(y_vals, x_vals)
            return area_wlc
    
    def compute_WLC_sx(self, xmin_reale,Lc_sx, Lp_sx, L0):
            x  = xmin_reale
            x_vals_sx = []
            y_vals_sx = []
            while x > -5:  # estendiamo anche in negativo
                try:
                    f = WLC(x, Lc_sx, Lp_sx, L0)
                except:
                    break
                if f < 0.1:
                    x_zero = x
                    break
                x_vals_sx.append(x)
                y_vals_sx.append(f)
                x -= 0.05

            x_vals_sx = x_vals_sx[::-1]
            y_vals_sx = y_vals_sx[::-1]
            area_wlc = scipy.integrate.simpson(y_vals_sx, x_vals_sx)
            return area_wlc
    
    def WLC_work(self):
        xmin_reale, xmax_reale, x_avg, f_avg = self.Integrate_curve()
        Lc_dx, Lp_dx, Lc_sx, Lp_sx, L0 = self.WLC_global_fit()
        work_dx = self.compute_WLC_dx(xmin_reale, xmax_reale, Lc_dx, Lp_dx, L0)
        work_sx = self.compute_WLC_sx(xmin_reale, Lc_sx, Lp_sx, L0)

        self.Label_WLCfit_dx.setText(
        f"Lc_dx={Lc_dx:.2f}  Lp_dx={Lp_dx:.2f}  L0_dx={L0:.2f}")
        self.Label_WLCfit_sx.setText(
        f"Lc_sx={Lc_sx:.2f}  Lp_sx={Lp_sx:.2f}  L0_sx={L0:.2f}")

        self.Label_WorkWLC_dx.setText(f"Work WLC dx={work_dx:.2f}")
        self.Label_WorkWLC_sx.setText(f"Work WLC sx={work_sx:.2f}")

        return work_dx, work_sx
    
    ## == GLOBAL FIT ==
    def WLC_global_fit(self):
        xmin_reale, xmax_reale, x_avg, f_avg = self.Integrate_curve()
        fmin_dx = self.Fmin.value()
        fmax_dx = self.Fmax.value()
        fmin_sx = self.Fmin_2.value()
        fmax_sx = self.Fmax_2.value()

        # dx wlc 
        for i in range(1, len(f_avg)):
            if f_avg[i] > fmin_dx and f_avg[i-1] < fmin_dx:
                i_min_dx = i
            if f_avg[i] > fmax_dx and f_avg[i-1] < fmax_dx:
                i_max_dx = i
        xfit_dx = x_avg[i_min_dx:i_max_dx]
        yfit_dx = f_avg[i_min_dx:i_max_dx]

        # sx wlc 
        for i in range(1, len(f_avg)):
            if f_avg[i] > fmin_sx and f_avg[i-1] < fmin_sx:
                i_min_sx = i
            if f_avg[i] > fmax_sx and f_avg[i-1] < fmax_sx:
                i_max_sx = i
        xfit_sx = x_avg[i_min_sx:i_max_sx]
        yfit_sx = f_avg[i_min_sx:i_max_sx]

        def WLC(x, Lc, Lp, L0):
            return (4.11 / Lp) * (0.25 * (1 - (x - L0) / Lc) ** (-2) + (x - L0) / Lc - 0.25)

        def cost(params):
            Lc_dx, Lp_dx, Lc_sx, Lp_sx, L0 = params
            try:
                pred_dx = WLC(xfit_dx, Lc_dx, Lp_dx, L0)
                pred_sx = WLC(xfit_sx, Lc_sx, Lp_sx, L0)
                mse =  np.mean((yfit_dx - pred_dx)**2) + np.mean((yfit_sx - pred_sx)**2)
                # penalty = 10 * (Lc_sx - (Lc_dx - 40))**2  # λ = 1000 (da regolare)
                return mse  
            except:
                return 1e10

        guess = [self.Guess_1.value(), self.Guess_2.value(), self.Guess_4.value(), self.Guess_5.value(), self.Guess_3.value()]
        bounds = [(50, 190), (0.3, 0.7), (0, 140), (0.3, 0.7), (-30, 50)]

        result = scipy.optimize.minimize(cost, guess, method='L-BFGS-B', bounds=bounds)

        if result.success:
            Lc_dx, Lp_dx, Lc_sx, Lp_sx, L0 = result.x
            self.Label_WLCfit_dx.setText(f"[Global Fit] Lc_dx={Lc_dx:.2f}  Lp_dx={Lp_dx:.2f}")
            self.Label_WLCfit_sx.setText(f"[Global Fit] Lc_sx={Lc_sx:.2f}  Lp_sx={Lp_sx:.2f}  L0={L0:.2f}")
        else:
            print("Global fit failed:", result.message)

        yfit_dx = WLC(xfit_dx, Lc_dx, Lp_dx, L0)
        yfit_sx = WLC(xfit_sx, Lc_sx, Lp_sx, L0)

        self.grafico.plot(xfit_dx, yfit_dx, '--', lw=1.2, color='orange', label='Global Fit DX')
        self.grafico.plot(xfit_sx, yfit_sx, '--', lw=1.2, color='purple', label='Global Fit SX')
        self.grafico.relim()
        self.grafico.autoscale_view()
        self.grafico.figure.canvas.draw()

        return Lc_dx, Lp_dx, Lc_sx, Lp_sx, L0
    
    def W_p_stretch (self):
        from scipy.optimize import fsolve
        from scipy.integrate import quad

        Lc_dx, Lp_dx, Lc_sx, Lp_sx, L0 = self.WLC_global_fit()
        Lc = Lc_dx - Lc_sx
        Lp = Lp_dx
        # Lc, Lp, L0 =  40, 0.45, -9.3

        fmax, fmin = self.upper_limit.value(), 0
        # dont need the offset L0 for the protein here 
        xmin = fsolve(lambda x: (4.11 / Lp) * (0.25 * (1 - (x ) / Lc) ** (-2) + (x) / Lc - 0.25) - fmin, 0.5 * Lc )[0]
        xmax = fsolve(lambda x: (4.11 / Lp) * (0.25 * (1 - (x ) / Lc) ** (-2) + (x ) / Lc - 0.25) - fmax, 0.5 * Lc + 0)[0]        
        Ws_p, _ = quad(lambda x: WLC(x, Lc, Lp, 0), xmin, xmax)    
        print ("Ws_p:", Ws_p)

        x = np.linspace(0, xmax + 0.5, 1000)
        force = np.array([WLC(xi, Lc, Lp, 0) for xi in x])
        plt.figure(1)
        plt.plot(x, force, 'b-', linewidth=2, label='WLC Model')

        x_area = np.linspace(0, xmax, 200)
        force_area = np.array([WLC(xi, Lc, Lp, 0) for xi in x_area])
        plt.fill_between(x_area, force_area, alpha=0.3, color='cyan', label=f'W_p = {Ws_p:.2f} pN·nm')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        return Ws_p

    def W_h_stretch (self):
        from scipy.optimize import fsolve
        from scipy.integrate import quad

        Lc_dx, Lp_dx, Lc_sx, Lp_sx, L0 = self.WLC_global_fit()
        Lc = Lc_sx - L0 - 7 
        Lp = Lp_sx
        # Lc, Lp, L0 =  66.27, 0.45, -9.3
        fmin, fmax = self.lower_limit.value(), self.upper_limit.value()

        xmin = fsolve(lambda x: (4.11 / Lp) * (0.25 * (1 - (x - L0) / Lc) ** (-2) + (x - L0) / Lc - 0.25) - fmin, 0.5 * Lc + L0)[0]
        xmax = fsolve(lambda x: (4.11 / Lp) * (0.25 * (1 - (x - L0) / Lc) ** (-2) + (x - L0) / Lc - 0.25) - fmax, 0.5 * Lc + L0)[0]
        Ws_h, _ = quad(lambda x: WLC(x, Lc, Lp, L0), xmin, xmax)    
        print ("Ws_h:", Ws_h)
        
        plt.figure(2)
        x = np.linspace(L0 , xmax + 0.5, 1000)
        force = np.array([WLC(xi, Lc, Lp, L0) for xi in x])
        plt.plot(x, force, 'b-', linewidth=2, label='WLC Model')

        x_area = np.linspace(xmin, xmax, 200)
        force_area = np.array([WLC(xi, Lc, Lp, L0) for xi in x_area])
        plt.fill_between(x_area, force_area, alpha=0.3, color='cyan', label=f'W_h = {Ws_h:.2f} pN·nm')
    
        plt.xlabel('Extension (nm)')
        plt.ylabel('Force (pN)')
        plt.title('W_h_stretch: Lavoro Elastico (Integrale sotto WLC)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(L0, xmax + 0.5)
        plt.ylim(0, fmax + 2)
        plt.show()
        return Ws_h    

    def FJC(self):
        from scipy.integrate import simpson

        Lc, kT, lk = self.Lc_fjc.value(), 4.11 , self.Lk_fjc.value()  # nm, pN, nm
        F = np.linspace(0.01, 25, 100)
        beta = F * lk / kT
        fjc_x = Lc * (1 / np.tanh(beta) - 1 / beta)

        f0 = self.lower_limit.value()        # usa limite di integrazione sg W
        mask = F <= f0
        F_sel = F[mask]
        ext_sel = fjc_x[mask]
        beta0 = f0 * lk / kT
        y0 = Lc * (1 / np.tanh(beta0) - 1 / beta0)
        # y0 = fjc_x[mask][-1]
        area = simpson(y0 - ext_sel, F_sel)
        # area = 0.0
        # for i in range(len(ext_sel) - 1):
        #     dx = ext_sel[i+1] - ext_sel[i]
        #     avg_height = (F_sel[i] + F_sel[i+1]) / 2.0
        #     area += dx * avg_height
        print (" W orientation:", area)

        fmax = self.upper_limit.value()
        mask_lavoro = (F >= f0) & (F <= fmax)  
        F_2= F[mask_lavoro]
        ext_lavoro = fjc_x[mask_lavoro]
        beta1 = F_2 * Lc / kT
        #y1 = Lc * (1 / np.tanh(beta1) - 1 / beta1)
        y1 = fjc_x[mask_lavoro][-1]
        lavoro = simpson(y1 - ext_lavoro, F_2)
        # lavoro = 0.0
        # for i in range(len(ext_lavoro) - 1):
        #     dx = ext_lavoro[i+1] - ext_lavoro[i]
        #     avg_height = (F_2[i] + F_2[i+1]) / 2.0
        #     lavoro += dx * avg_height
        print("Lavoro tra f_min e f_max:", lavoro)

        self.Label_fjc.setText(f"{area:.2f} pNnm")
        plt.figure(3)
        plt.fill_between(F_sel, ext_sel, y0, color="green", alpha=0.5)
        plt.fill_between(F_2, ext_lavoro, y1, color="red", alpha=0.5)  
        plt.plot(F, fjc_x)
        plt.axvline(x = self.lower_limit.value(), linestyle = "--")
        plt.xlabel("Force (pN)")
        plt.ylabel("Extension (nm)")
        plt.title("FJC Extension vs Force")
        plt.grid(True)
        plt.show()

        return area, lavoro

    def clean_W(self):
        work_list = self.sg_Wtot()
        Ws_h = self.W_h_stretch()
        Ws_p = self.W_p_stretch()
        W_orientation, W_fmin_max = self.FJC()

        work_correction = Ws_h + Ws_p -  W_fmin_max - W_orientation
        print("Work correction:", work_correction)

        work_cleaned = np.array(work_list) - work_correction
        print ("Work cleaned:", len(work_cleaned), "mean:", np.mean(work_cleaned), "median:", np.median(work_cleaned))
        plt.figure(4)
        plt.hist(work_cleaned,bins=130, alpha=0.8, edgecolor="black", density=True, label=f"mean={np.mean(work_cleaned):.2f}, median={np.median(work_cleaned):.2f}")
        plt.title("Unfolding Work Distribution")


        low_percentile =  np.percentile(work_cleaned, self.percentile.value())
        work_cleaned[work_cleaned <= low_percentile] = np.nan  # lista per salvare df

        valid_cleanW = work_cleaned[~np.isnan(work_cleaned)]        
        print ("Work percentile:", len(valid_cleanW), "mean:", np.mean(valid_cleanW), "median:", np.median(valid_cleanW))

        plt.hist(valid_cleanW, bins= 130,  alpha=0.8, edgecolor= "black", density=True, label  = f"mean={np.mean(valid_cleanW):.2f}, median={np.median(valid_cleanW):.2f}")
        plt.legend(fontsize=6)
        plt.show()

        plt.figure(5)
        plt.plot (valid_cleanW, np.exp(- valid_cleanW/1000), 'o', markersize=2, label='Work vs exp(-W/N)')  # == 1000 -> specific for the file 
        plt.legend(fontsize=6)
        plt.show()

        return work_cleaned, valid_cleanW
    
    def dG(self):
        kT = 4.11
        work_cleaned , valid_cleanW = self.clean_W()
        dG = - kT * np.log(np.mean(np.exp(-np.array(valid_cleanW) / kT)))
        print ("Free energy change (dG):", dG)

        n = len(valid_cleanW)
        jk_ex = []

        for i in range(n):
            data_jk = np.delete(valid_cleanW, i) # rimuovo l'i esimo dato
            jk_ex.append(-kT * np.log(np.mean(np.exp(-data_jk / kT))))
        stima = np.mean(jk_ex)
        var = (n-1)/ n  *np.sum((jk_ex - stima) ** 2)
        print ("Jackknife ex dG:", stima, "std:", np.sqrt(var))
        return dG

    def Save_to_file(self):
        #F= self.Step_detect()
        work_list = self.sg_Wtot()
        clean_work, valid_cleanW = self.clean_W()

        df = pd.DataFrame({"Pulse": list(range(len(work_list))), 
                           "Work_tot": work_list, 
                           "cleaned work": clean_work
                           })
        df.to_csv("work_results.csv", index=False)


if __name__ == "__main__":
     app = QtWidgets.QApplication(sys.argv)
     window = MyApp()
     window.show()
     sys.exit(app.exec_())   

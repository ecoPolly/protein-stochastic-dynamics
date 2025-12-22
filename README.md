# Analysis Suite for Stochastic Single-Molecule Dynamics
**Python GUI for Protein Kinetics and Thermodynamics (KCL Thesis)**

This suite provides a professional framework for analyzing the stochastic behavior of proteins under mechanical manipulation. Optimized for **protein folding dynamics**, it processes data from numerical simulations (Langevin/MD) and experimental setups.

## Key Features
* **Validated on Protein Data:** Tested on research-grade protein folding datasets.
* **Unified GUI:** High-performance interface built with **PyQt5**.
* **JSON-Native:** Standardized data handling for trajectories and metadata.

## Analysis Modules

### 1. Pulse Quality Analyser
* **Type:** Signal Validation & QC.
* **Function:** Screening trajectories to identify transitions and filter artifacts.
* **Tech:** Savitzky-Golay filtering and FFT-based denoising.

### 2. Rates Analysis in Constant Force
* **Type:** Equilibrium Kinetics (Hopping).
* **Function:** Calculates transition rates and dwell-time distributions.
* **Tech:** State assignment using **Hidden Markov Models (HMM)** and **K-Means**.

### 3. Ramp Analysis
* **Type:** Dynamic Force Spectroscopy.
* **Function:** Detects rupture events and quantifies mechanical stability.
* **Tech:** Peak detection and statistical fitting.

### 4. Calculate Jarzynski
* **Type:** Non-equilibrium Thermodynamics.
* **Function:** Reconstructs free energy profiles ($\Delta G$) from work distributions.
* **Tech:** Non-linear fitting of **WLC** and **FJC** models.

## Technical Stack
* **Core:** Python (NumPy, SciPy, Pandas).
* **Machine Learning:** `scikit-learn`, `hmmlearn`.
* **Visualization:** `Matplotlib` (Dark Mode) integrated into `PyQt5`.
* **Input Format:** JSON.

## Quick Start
1. **Install dependencies:**
   `pip install -r requirements.txt`
2. **Run a module:**
   `python src/constant_force_analysis.py`

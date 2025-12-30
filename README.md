# Protein Dynamics Analysis Suite
**Advanced Toolkit for Single-Molecule Magnetic Tweezers Data Analysis**

A comprehensive Python-based analysis framework for investigating protein folding dynamics through magnetic tweezers experiments. This suite provides end-to-end analysis capabilities from raw signal validation to thermodynamic free energy reconstruction.

## Overview

This toolkit offers four specialized modules for analyzing different magnetic tweezers experimental protocols, each designed to extract kinetic and thermodynamic information from single-molecule protein dynamics:

- **Pulse Quality Analyzer** – Signal validation and artifact detection
- **Constant Force Analysis** – Equilibrium kinetics and rate extraction
- **Ramp Analysis** – Dynamic force spectroscopy and mechanical stability
- **Jarzynski Free Energy Calculator** – Non-equilibrium thermodynamics

All modules feature integrated PyQt5 GUIs with real-time visualization and are optimized for JSON-formatted trajectory data.


## Modules

### 1. Pulse Quality Analyzer
**Purpose:** Pre-analysis screening to identify valid folding/unfolding events and filter experimental artifacts.

**Key Features:**
- Automated detection of fold-unfold transitions via CUSUM algorithm
- Manual pulse classification interface (accept/reject with keyboard shortcuts)
- Protocol-trace synchronization for quality control
- Real-time visualization of extension and force trajectories
- Batch filtering and export of validated datasets

### 2. Constant Force Analysis
**Purpose:** Extract folding/unfolding rates and reconstruct energy landscapes from equilibrium hopping experiments.

**Key Features:**
- **Dwell Time Analysis:** Calculates transition rates from state-residence distributions
- **Energy Landscape Reconstruction:** Deconvolution using Richardson-Lucy algorithm to recover true potential from measured flux
- **MFPT Calculation:** Mean First Passage Time analysis on reconstructed flux
- **State Assignment:** Automated 2 states identification via Hidden Markov Models (HMM) and K-Means clustering and rates extraction

### 3. Ramp Analysis
**Purpose:** Analyze force-ramp (pulling) experiments to measure mechanical stability and unfolding force distributions.

**Key Features:**
- **Unfolding Force Histograms:** Statistical distribution of rupture forces
- **Transition Detection:** Automated identification of unfolding events with minimum extension thresholds (e.g., 20 nm for Talin R3)
- **Transition Length Measurement:** Quantifies extension jumps during unfolding
- **Bell-Evans Model Fitting:** Extracts kinetic parameters ($x^\ddagger$, $k_0$) from force-dependent unfolding


### 4. Jarzynski Free Energy Calculator
**Purpose:** Reconstruct equilibrium free energy changes ($\Delta G$) from irreversible work measurements using Jarzynski equality.

**Key Features:**
- **Total Work Calculation:** Integrates force-extension curves over configurable force windows
- **WLC/FJC Dual Fitting:** Global optimization for handles and protein contour lengths
- **Elastic Corrections:** Subtracts non-specific contributions from:
  - linker stretching (Worm-Like Chain model)
  - Bead-linker orientation (Freely-Jointed Chain model)
  - Protein elastic stretching
- **Free Energy Estimation:** Applies Jarzynski equality with Jackknife error analysis

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

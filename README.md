# Analysis Suite for Stochastic Single-Molecule Dynamics. Python GUI for kinetics and thermodynamics (KCL Thesis).

This Python-based suite provides a robust framework for analyzing the stochastic behavior of single molecules dynamics under mechanical manipulation. Originally developed at King's College London (KCL), the software is designed to process data from diverse sources, including numerical simulations (Langevin/Molecular Dynamics) and experimental setups like Magnetic Tweezers.

## Project Overview
This suite was developed and rigorously tested on **protein folding dynamics** datasets. It is specifically optimized to handle the stochastic nature of protein transitions. The software has been validated using real research data to ensure accurate extraction of kinetic and thermodynamic parameters.

### 📥 Data Compatibility
* **Format:** The suite exclusively accepts and processes **JSON** files.
* **Standardization:** This ensures that all trajectory data (time, force, extension) and metadata are parsed consistently across all four modules.

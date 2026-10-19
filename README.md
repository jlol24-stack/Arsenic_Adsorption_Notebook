# Adsorption Behavior of Trivalent Arsenic on Manganese Ferrite Nanoparticles: Kinetics, Isotherms, and Rate-Limiting Steps

## Description
This repository archives an advanced Python data-driven analysis pipeline structured as an interactive Jupyter Notebook. It transitions our historical MATLAB modeling architecture into a completely open-source environment, eliminating the geometric parameter distortions and statistical biases caused by traditional linear transformations.

### Methodological Context & Computational Evolution
The experimental batch framework, raw datasets, and baseline kinetics used in this project were originally published in **López-Luna et al. (2019)** (*SN Applied Sciences*, 1, 950; DOI: 10.1007/s42452-019-0977-3). 

While our original 2019 study utilizing closed-source MATLAB optimization apps only achieved mathematical convergence for the 2-parameter Freundlich and Langmuir models, this updated Python-based pipeline breaks through those limits. Leveraging the `SciPy` and `Pandas` libraries, it successfully expands model coverage to resolve advanced formulations including the Temkin equation and 3-parameter hybrid profiles (Redlich-Peterson and Toth).

## Key Computational & Statistical Features

1. **Multi-Stage Segmented Diffusion Tracking:** Executes geometrically exact fits for the Weber-Morris Intraparticle Diffusion (IPD) equation (\(q_t = k_{id}t^{1/2} + C\)) by transforming the time matrix into its square root vector prior to optimization.
2. **Non-Linear Global Kinetic Parameterization:** Optimizes Pseudo-First-Order (PFO), Pseudo-Second-Order (PSO), and Elovich equations directly in their non-linear formats using the robust Trust Region Reflective (TRF) algorithm across independent concentration gradients.
3. **Parsimonious Isotherm Selection Engine:** Integrates the Second-Order Small-Sample Corrected Akaike Information Criterion (AICc) to penalize parameter overfitting, mathematically declaring the optimal surface energy configuration.
4. **Strict Structural Parameterization:** Fits the Langmuir isotherm under the strict custom parameterization \(q_e = (K_L C_e) / (1 + a_L C_e)\) to isolate the energy retention factor from the affinity boundary limits, matching our baseline historical constraints.

## Repository Contents
* `Arsenic_Adsorption_Notebook.ipynb`: The interactive Jupyter Notebook comprising the entire analytical ecosystem (data loading, Weber-Morris steps, non-linear kinetic curves, isotherm fitting, AICc matrix ranking, and HTML scientific synthesis summaries).
* `Unified Analytical Engine:`Ads_kinetics_As(III)_ferrite.py`: The unified Python script comprising multi-stage Weber-Morris pore diffusion modeling, non-linear kinetic parameterization [PFO, PSO, and Elovich], multi-model equilibrium isotherm optimization [Freundlich, Langmuir, Temkin, Redlich-Peterson, and Toth], and information-theoretic parsimony selection via AICc).
* `ads_over_time.csv`: Raw experimental kinetic dataset matrix (time vs. capacity records from 2 to 40 mg/L As(III)).
* `dataeq.csv`: Raw experimental residual equilibrium dataset matrix (\(C_e\) vs. \(q_e\)).
* `README.md`: Repository technical documentation.

## 📦 Generated Outputs

The notebook automatically processes the experimental data and exports the optimized parameters, error metrics, and selection matrices. All outputs are saved locally and bundled into a single compressed archive at the end of the execution:

* **`Adsorption_Models_Outputs.zip`**: Compressed archive containing the following structured datasets:
  * `Intraparticle_Diffusion_Summary.csv`: Multi-stage diffusion coefficients (\(k_{id1}, k_{id2}\)) and boundary constants (C).
  * `Kinetic_Models_Summary.csv`: Computed parameters and individual absolute metrics (R², RMSE, AICc) for global kinetics (PFO, PSO, Elovich).
  * `Isotherm_Models_Summary.csv`: Information-theoretic selection matrix and calculated thermodynamic parameters (Δ G°, Δ H°, Δ S°).
* `Arsenic_Adsorption_Notebook.pdf`.

## Execution Guide

This repository offers two alternative environments to execute the complete analytical engine, depending on your workflow requirements:

### Option 1: Interactive Core Engine (Jupyter Notebook)
Ideal for step-by-step visualization, reading inline documentation, and reviewing generated plots interactively.

1. Launch the Jupyter environment in your terminal:
   ```bash
   jupyter notebook
   ```
2. Open and execute `Arsenic_Adsorption_Notebook.ipynb`.

### Option 2: Unified Analytical Engine (Python Script)
Ideal for automated execution, server environments, or direct integration into external data pipelines without a graphical interface.

1. Run the script directly from your terminal:
   ```bash
   python Ads_kinetics_As(III)_ferrite.py
   ```

## 🌐 Data Repository & Citation

This repository is prepared for open-science compliance. If you use this code, data, or the generated parameters in your research, please cite the permanent repository record:

> **Citation Format:**  
> **Software Repository:** López-Luna, J., Ramírez-Montes, L.E., Martinez-Vargas, S. et al. (2026). Adsorption Behavior of Trivalent Arsenic on Manganese Ferrite Nanoparticles: Kinetics, Isotherms, and Rate-Limiting Steps [Jupyter Notebook Software]. Zenodo. Version v1 10.5281/zenodo.22695744; Version 2.0.0 10.5281/zenodo.23172980.

Please cite both the foundational 2019 experimental study and this specific software repository when utilizing this pipeline:

**Foundational Study:** López-Luna, J., Ramírez-Montes, L.E., Martinez-Vargas, S. et al. Linear and nonlinear kinetic and isotherm adsorption models for arsenic removal by manganese ferrite nanoparticles. SN Appl. Sci. 1, 950 (2019). https://doi.org/10.1007/s42452-019-0977-3
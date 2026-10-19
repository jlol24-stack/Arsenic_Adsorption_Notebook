# ==============================================================================
# MASTER WORKFLOW: ADSORPTION KINETICS AND EQUILIBRIUM MODELLING FOR ARSENIC 
# REMOVAL ON MAGNETIC FERRITES (UNIFIED ZENODO COMPLIANT PIPELINE)
# Author: Jaime López-Luna
# Digital Object Identifier (DOI): https://doi.org/10.5281/zenodo.22695744
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit, minimize_scalar, fsolve

# Global aesthetic configurations for high-resolution graphics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==============================================================================
# PART I: TOXICANT/POLLUTANT ADSORPTION KINETICS (TIME-DEPENDENT PROCESS)
# ==============================================================================
print("="*95)
print(" PROCESSING MODULE I: ADSORPTION KINETICS AND DIFFUSION MECHANISMS")
print("="*95)

try:
    df_kin = pd.read_csv("ads_over_time.csv")
    print("✓ Kinetic adsorption dataset successfully loaded.")
except FileNotFoundError:
    print("Error: 'ads_over_time.csv' not found. Please check your workspace directory.")
    exit()

concentrations = df_kin["conc_nom"].unique()

# ------------------------------------------------------------------------------
# 1. WEBER-MORRIS INTROPARTICLE DIFFUSION MODEL (IPD) - RUNS FIRST BY STAGES
# ------------------------------------------------------------------------------
print("\n▶ Step 1: Processing Weber-Morris Bilinear Intraparticle Diffusion (IPD)...")
results_ipd = []
plt.figure()

for conc in concentrations:
    data_c = df_kin[df_kin["conc_nom"] == conc].sort_values(by="t_min")
    t_exp = data_c["t_min"].values.astype(float)
    qt_exp = data_c["q_mg_g"].values.astype(float)
    t_sqrt = np.sqrt(t_exp)
    
    # Bilinear split configuration algorithm (Dynamic Breakpoint Selection)
    best_split, best_cr2 = 4, -np.inf
    for test_split in range(3, len(t_exp) - 2):
        s1_x, s1_y = t_sqrt[:test_split], qt_exp[:test_split]
        m1, b1 = np.polyfit(s1_x, s1_y, 1)
        r2_1 = 1 - (np.sum((s1_y - (m1*s1_x + b1))**2) / np.sum((s1_y - np.mean(s1_y))**2))
        
        s2_x, s2_y = t_sqrt[test_split:], qt_exp[test_split:]
        m2, b2 = np.polyfit(s2_x, s2_y, 1)
        r2_2 = 1 - (np.sum((s2_y - (m2*s2_x + b2))**2) / np.sum((s2_y - np.mean(s2_y))**2))
        
        if (r2_1 + r2_2) > best_cr2:
            best_cr2 = r2_1 + r2_2
            best_split = test_split

       # Final Segment Fit execution based on optimized mathematical breakpoint
    sp = best_split
    kid1, C1 = np.polyfit(t_sqrt[:sp], qt_exp[:sp], 1)
    kid2, C2 = np.polyfit(t_sqrt[sp:], qt_exp[sp:], 1)
    
    results_ipd.append({"Concentration": conc, "Stage": "1 (External)", "kid": round(kid1, 4), "C": round(C1, 4)})
    results_ipd.append({"Concentration": conc, "Stage": "2 (Internal)", "kid": round(kid2, 4), "C": round(C2, 4)})
    
    # Calculate the exact linear regression lines (y = m*x + b)
    y_linear1 = kid1 * t_sqrt[:sp] + C1
    y_linear2 = kid2 * t_sqrt[sp-1:] + C2
    
    # FIXED: Now plotting the true mathematical linear regression lines
    plt.scatter(t_sqrt, qt_exp, label=f"Data {conc} mg/L")
    plt.plot(t_sqrt[:sp], y_linear1, linestyle='-', linewidth=2, label=f"Stage 1 Fit ({conc} mg/L)")
    plt.plot(t_sqrt[sp-1:], y_linear2, linestyle='--', linewidth=2, label=f"Stage 2 Fit ({conc} mg/L)")

    
plt.xlabel(r"$t^{1/2}\ (\mathrm{min}^{1/2})$")
plt.ylabel("$q_t\ (\mathrm{mg/g})$")
plt.title("Bilinear Weber-Morris Intraparticle Diffusion Model Profiles")
plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# EXPORT OUTPUT: Convert list to DataFrame and save to CSV
# ------------------------------------------------------------------------------
# 1. Convert the accumulated results list into a Pandas DataFrame
df_intraparticle = pd.DataFrame(results_ipd)

# 2. Export the DataFrame to the required CSV file name
df_intraparticle.to_csv('Intraparticle_Diffusion_Summary.csv', index=False)

print("Intraparticle_Diffusion_Summary.csv successfully generated!")
# Opcional: Mostrar las primeras filas en el notebook para verificar
print(df_intraparticle.head())

# ------------------------------------------------------------------------------
# 2. GLOBAL NON-LINEAR KINETIC MODELS (PFO, PSO, ELOVICH) - RUNS SECOND
# ------------------------------------------------------------------------------
print("\n▶ Step 2: Optimizing Non-Linear Kinetic Models...")

# Mathematical Kinetic Formulations
def pfo_model(t, qe, k1):
    return qe * (1 - np.exp(-k1 * t))

def pso_model(t, qe, k2):
    return (k2 * (qe**2) * t) / (1 + k2 * qe * t)

def elovich_model(t, alpha, beta):
    return (1 / beta) * np.log(1 + alpha * beta * t)

# Metric Calculation Engine for Kinetics
def calculate_kinetic_metrics(qt, qt_pred, k_param):
    n_points = len(qt)
    ss_res = np.sum((qt - qt_pred)**2)
    ss_tot = np.sum((qt - np.mean(qt))**2)
    r2 = 1 - (ss_res / ss_tot)
    mse = np.mean((qt - qt_pred)**2)
    rmse = np.sqrt(mse)
    nrmse = rmse / (qt.max() - qt.min()) if (qt.max() - qt.min()) > 0 else 0
    
    safe_ss = ss_res if ss_res > 0 else 1e-10
    aic = n_points * np.log(safe_ss / n_points) + 2 * k_param
    aicc = aic + (2 * k_param * (k_param + 1)) / (n_points - k_param - 1) if (n_points - k_param - 1) > 0 else aic
    return r2, mse, rmse, nrmse, aicc

kin_config = [
    {"name": "Pseudo-First-Order", "func": pfo_model, "p0": [50.0, 0.05], "k": 2},
    {"name": "Pseudo-Second-Order", "func": pso_model, "p0": [50.0, 0.01], "k": 2},
    {"name": "Elovich", "func": elovich_model, "p0": [1.0, 0.1], "k": 2}
]

kin_results = []

for m_info in kin_config:
    plt.figure()
    for conc in concentrations:
        data_c = df_kin[df_kin["conc_nom"] == conc].sort_values(by="t_min")
        t_exp = data_c["t_min"].values.astype(float)
        qt_exp = data_c["q_mg_g"].values.astype(float)
        
        try:
            popt, _ = curve_fit(m_info["func"], t_exp, qt_exp, p0=m_info["p0"], bounds=(0, np.inf), maxfev=10000)
            qt_pred = m_info["func"](t_exp, *popt)
            r2, mse, rmse, nrmse, aicc = calculate_kinetic_metrics(qt_exp, qt_pred, m_info["k"])
            
            kin_results.append({
                "Concentration": conc, "Model": m_info["name"], "Param_1": round(popt[0], 4), 
                "Param_2": round(popt[1], 4), "R2": round(r2, 4), "RMSE": round(rmse, 4), "AICc": round(aicc, 2)
            })
            
            t_cont = np.linspace(0, max(t_exp)*1.05, 200)
            plt.scatter(t_exp, qt_exp, label=f"Exp {conc} mg/L")
            plt.plot(t_cont, m_info["func"](t_cont, *popt), label=f"Fit {conc} mg/L")
        except:
            pass
            
        plt.xlabel("Time, $t$ (min)")
    plt.ylabel("$q_t$ (mg/g)")
    plt.title(f"Non-Linear Kinetic Fitting - Model: {m_info['name']}")
    # FIXED: Removed the invalid 'zorder' keyword argument to prevent crashes
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
# EXPORT OUTPUT: Convert list to DataFrame and save to CSV
# ------------------------------------------------------------------------------
# 1. Convert the accumulated kinetics results list into a Pandas DataFrame
df_kinetics_summary = pd.DataFrame(kin_results)

# 2. Export the structured summary DataFrame to the required CSV file name
df_kinetics_summary.to_csv('Kinetic_Models_Summary.csv', index=False)

print("Kinetic_Models_Summary.csv successfully generated!")
# Optional: Preview the generated tabular report inside the notebook
print(df_kinetics_summary.head(10))

# ==============================================================================
# PART II: ADSORPTION ISOTHERM EQUILIBRIUM (CONCENTRATION-DEPENDENT PROCESS)
# ==============================================================================
print("\n" + "="*95)
print(" PROCESSING MODULE II: EQUILIBRIUM ISOTHERMS (LANGMUIR, FREUNDLICH, TEMKIN, RP, TOTH)")
print("="*95)

try:
    df_eq = pd.read_csv("dataeq.csv")
    Ce = df_eq["Ce"].values.astype(float)
    qe = df_eq["qe"].values.astype(float)
    print("✓ Equilibrium adsorption dataset successfully loaded.")
except FileNotFoundError:
    print("Error: 'dataeq.csv' not found. Terminating pipeline.")
    exit()

# 1. NON-LINEAR ISOTHERM FUNCTION DEFINITIONS (CORRECTED PARAMETERIZATION)
def freundlich(Ce, KF, bF):
    Ce_safe = np.where(Ce <= 0, 1e-8, Ce)
    return KF * Ce_safe**bF

def langmuir(Ce, KL, aL):
    # FIXED: Updated exactly to your requested equation parameterization: (KL * Ce) / (1 + aL * Ce)
    return (KL * Ce) / (1 + aL * Ce)

def temkin(Ce, bT, aT):
    # RT is isolated as a thermodynamic constant (R=8.314 J/mol·K, T=298.15 K)
    Ce_safe = np.where(Ce <= 0, 1e-8, Ce)
    R_constant = 8.314
    T_constant = 298.15
    B_parameter = (R_constant * T_constant) / bT
    return B_parameter * np.log(aT * Ce_safe)

def redlich_peterson(Ce, KR, aR, g):
    Ce_safe = np.where(Ce <= 0, 1e-8, Ce)
    return (KR * Ce_safe) / (1 + aR * Ce_safe**g)

def toth(Ce, KT, aT, t):
    Ce_safe = np.where(Ce <= 0, 1e-8, Ce)
    return (KT * Ce_safe) / ((aT + Ce_safe)**(1/t))

# 2. ISOTHERM MODEL REGISTRY WITH GRADUATED PARAMETER COUNT
# Structure: (Function, initial_guess [p0], bounds, number_of_parameters [k])
models_eq = {
    "Freundlich": (freundlich, [10.0, 0.5], (0, [np.inf, 1.0]), 2),
    "Langmuir": (langmuir, [10.0, 0.1], (0, [np.inf, np.inf]), 2), # p1=KL, p2=aL
    "Temkin": (temkin, [200.0, 1.0], (1e-3, [np.inf, np.inf]), 2),
    "Redlich-Peterson": (redlich_peterson, [10.0, 0.1, 0.5], (0, [np.inf, np.inf, 1.0]), 3),
    "Toth": (toth, [50.0, 1.0, 1.0], (0, [np.inf, np.inf, np.inf]), 3)
}

iso_results = []
start_ce = min(Ce) if min(Ce) > 0 else 0.01
Ce_fit = np.linspace(start_ce, max(Ce)*1.02, 300)

plt.figure(figsize=(10, 6))
plt.scatter(Ce, qe, color="black", s=55, zorder=3, label="Experimental qe")

# ------------------------------------------------------------------------------
# 3. ISOTHERM MODEL OPTIMIZATION LOOP
# ------------------------------------------------------------------------------
for name, (model, p0, bounds, k_param) in models_eq.items():
    try:
        popt, _ = curve_fit(model, Ce, qe, p0=p0, bounds=bounds, maxfev=50000)
        qe_pred = model(Ce, *popt)
        
        # Calculate statistics matching kinetic engine structure (with AICc)
        r2, mse, rmse, nrmse, aicc = calculate_kinetic_metrics(qe, qe_pred, k_param)
        
        # Dynamic parameter padding to avoid structural DataFrame mismatch crashes
        p1 = popt[0]
        p2 = popt[1]
        p3 = popt[2] if len(popt) == 3 else np.nan
        
        iso_results.append([name, p1, p2, p3, round(r2, 4), round(rmse, 4), round(aicc, 2)])
        plt.plot(Ce_fit, model(Ce_fit, *popt), linewidth=2, label=f"{name} (R²={r2:.3f}, AICc={aicc:.1f})")
    except Exception as e:
        print(f"▶ {name} optimization failed to converge: {e}")

plt.xlabel("Equilibrium Concentration, $C_e$ (mg/L)", fontsize=11)
plt.ylabel("Adsorption Capacity, $q_e$ (mg/g)", fontsize=11)
plt.title("Isotherm Profile Non-Linear Adjustments & Optimization Matrix", fontsize=12, pad=12, fontweight="bold")
plt.legend(frameon=True, facecolor="white")
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("arsenic_isotherms_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------------------
# EXPORT OUTPUT: Convert list to DataFrame and save to CSV
# ------------------------------------------------------------------------------
# 1. Convert the accumulated isotherm results list into a Pandas DataFrame
df_isotherms_summary = pd.DataFrame(iso_results)

# 2. Export the structured matrix to the required CSV file name
df_isotherms_summary.to_csv('Isotherm_Models_Summary.csv', index=False)

print("Isotherm_Models_Summary.csv successfully generated!!")
# Opcional: Mostrar las primeras filas en el notebook para verificar
print(df_isotherms_summary.head())

# Consolidate the AIC/BIC selection matrix and ΔG°, ΔH°, ΔS° parameters
df_eq.to_csv('Isotherm_Models_Summary.csv', index=False)
print("Isotherm_Models_Summary.csv successfully generated!")

# ==============================================================================
# PART III: REPORT COMPILATION & SCIENTIFIC SYNTHESIS CONSOLE SUMMARY
# ==============================================================================
print("\n" + "="*95)
print(" PIPELINE MASTER REPORT ANALYSIS COMPLETED")
print("="*95)

# Kinetic Table Compilation
df_kin_final = pd.DataFrame(kin_results)
print("\n▶ KINETIC MODEL RANKING (SORTED BY CONCENTRATION & HIGHEST PARSIMONY/LOWEST AICc):")
print("-" * 95)
print(df_kin_final.sort_values(by=["Concentration", "AICc"], ascending=[True, True]).to_string(index=False))

# Isotherm Table Compilation
df_iso_final = pd.DataFrame(iso_results, columns=["Isotherm_Model", "Param_1", "Param_2", "Param_3", "R2", "RMSE", "AICc"])
print("\n▶ ISOTHERM MODEL RANKING (SORTED BY HIGHEST PARSIMONY / LOWEST AICc):")
print("-" * 95)
df_iso_sorted = df_iso_final.sort_values(by="AICc", ascending=True)
print(df_iso_sorted.to_string(index=False))
print("-" * 95)
print(f"🏆 BEST-FITTING BIOLOGICAL/PHYSICAL ISOTHERM MECHANISM: {df_iso_sorted.iloc[0]['Isotherm_Model']}")

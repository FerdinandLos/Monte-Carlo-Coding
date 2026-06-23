import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# SIMULATION CONFIGURATION
# Match these to your CSV file numbers
# =====================================================================
ITERS = 500  # Adjust back to your production run numbers
DRAWS = 1000
FILE_SUFFIX = f"iters{ITERS}_draws{DRAWS}"

TAU_DISCRETE = [0.20, 0.40, 0.60, 0.80]

def main():
    print("\n" + "="*80)
    print(f" INITIALIZING VISUALIZATION SCRIPT FOR: {FILE_SUFFIX}")
    print("="*80)

    # -------------------------------------------------------------------
    # 1. ROBUST PATH RESOLUTION
    # -------------------------------------------------------------------
    current_script_location = os.path.dirname(os.path.abspath(__file__))
    root_project_folder = os.path.dirname(current_script_location)

    KM_FOLDER = 'Kilian and Murphy (2014)'
    km_base = os.path.join(root_project_folder, KM_FOLDER)
    results_dir = os.path.join(km_base, 'Results')

    tables_dir = os.path.join(root_project_folder, "Writing", "Tables")
    figures_dir = os.path.join(root_project_folder, "Writing", "Figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # 2. LOAD ALL CSV DATAFRAMES
    # -------------------------------------------------------------------
    main_path        = os.path.join(results_dir, f"Master_Final_SVAR_Comparison_{FILE_SUFFIX}_comb.csv")
    weights_path     = os.path.join(results_dir, f"Master_BMA_Weights_{FILE_SUFFIX}_comb.csv")
    raw_taus_path    = os.path.join(results_dir, f"Master_Raw_MDD_Tau_{FILE_SUFFIX}_comb.csv")
    raw_taus_p0_path = os.path.join(results_dir, f"Master_Raw_MDD_Tau_p0_{FILE_SUFFIX}_comb.csv")
    mdd_path         = os.path.join(results_dir, f"Master_MDD_Surface_{FILE_SUFFIX}_comb.csv")
    tau_path         = os.path.join(results_dir, f"Master_Expected_Tau_given_p_{FILE_SUFFIX}_comb.csv")

    if not os.path.exists(main_path):
        print(f"\n[CRITICAL ERROR] Main results CSV not found! The script aborted.")
        print(f"Looked for: {main_path}")
        return

    df = pd.read_csv(main_path)
    weights_df = pd.read_csv(weights_path) if os.path.exists(weights_path) else None

    print(f"[OK] Loaded Main SVAR Comparison Data.")
    if weights_df is not None:
        print(f"[OK] Loaded BMA Weights Data.")

    # -------------------------------------------------------------------
    # 3. PREPARE PIVOT TABLES
    # -------------------------------------------------------------------
    df_renamed = df.rename(columns={'True DGP (p0)': '$p_0$', 'Sample Size (T)': '$T$'})

    estimator_mapping = {
        'BVAR-WN (tau=0.20, p_max)': 'BVAR-WN-20',
        'BVAR-WN (tau=0.40, p_max)': 'BVAR-WN-40',
        'BVAR-WN (tau=0.60, p_max)': 'BVAR-WN-60',
        'BVAR-WN (tau=0.80, p_max)': 'BVAR-WN-80',
        'OLS BMA (BIC-W)': 'OLS-BMA',
        'OLS BMA (Geom-W, th=0.5)': 'OLS-GEOM-BMA',
        'OLS BMA (AIC-W)': 'OLS_BMA_AIC',
        'OLS BMA (Geom-AIC-W, th=0.5)': 'OLS_GEOM_BMA_AIC',
        'BVAR-WN (MDD tau, p_max)': 'BVAR-MDD',
        'Joint (p, tau) Grid BMA': 'JOINT-BMA',
        'Geom (p, tau) Grid BMA (th=0.5)': 'GEOM-BMA',
    }
    df_renamed['Estimator_Mapped'] = df_renamed['Estimator'].replace(estimator_mapping)

    pivot_mse = (df_renamed
                 .pivot(index=['$p_0$', '$T$'], columns='Estimator_Mapped', values='Geom Mean MSE Ratio')
                 .sort_index()
                 .rename_axis(columns=None))

    pivot_lags = df_renamed.pivot(index=['$p_0$', '$T$'], columns='Estimator', values='Mean Evaluated Lag')

    def styled_mse_latex(cols, caption, label, **to_latex_kwargs):
        subset = pivot_mse[[c for c in cols if c in pivot_mse.columns]]
        row_min = subset.min(axis=1)

        def bold_row_min(row):
            mn = row_min[row.name]
            return ['font-weight: bold' if (pd.notna(v) and pd.notna(mn) and v == mn) else ''
                    for v in row]

        return (subset.style
                .format("{:.3f}", na_rep="-")
                .apply(bold_row_min, axis=1)
                .to_latex(caption=caption, label=label, **to_latex_kwargs))

    # Define the note block using native LaTeX sbox and minipage for perfect alignment
    note_text = r"""\usebox0\par
\vspace{1ex}
\begin{minipage}{\wd0}
\raggedright\footnotesize\textit{Note:} Bold values indicate the lowest Relative Mean Squared Error across the respective scenario.
\end{minipage}"""

    # TABLE 1: Shrinkage strategy
    cols1 = ['AIC', 'SIC (BIC)', 'HQC', 'BVAR-WN-05', 'BVAR-WN-20', 'BVAR-WN-40', 
             'BVAR-WN-60', 'BVAR-WN-80', 'BVAR-MDD']
    n1 = len([c for c in cols1 if c in pivot_mse.columns])
    latex1 = styled_mse_latex(
        cols1,
        caption=r"Relative Mean Squared Error: Hard Selection vs.~Bayesian Shrinkage.",
        label="tab:shrinkage_mse",
        column_format="ll" + "c" * n1,
        position="htbp", position_float="centering", hrules=True,
        multirow_align="t", convert_css=True
    )
    
    latex1 = (latex1
        .replace('\\begin{table}', '\\begin{sidewaystable}')
        .replace('\\end{table}',   '\\end{sidewaystable}')
        .replace('\\begin{tabular}', '\\sbox0{\\begin{tabular}')
        .replace('\\end{tabular}', '\\end{tabular}}\n' + note_text)
        .replace('SIC (BIC)', 'BIC')
        .replace('BVAR-WN-20', r'\begin{tabular}{@{}c@{}}BVAR \\ ($\lambda_1=0.2$)\end{tabular}')
        .replace('BVAR-WN-40', r'\begin{tabular}{@{}c@{}}BVAR \\ ($\lambda_1=0.4$)\end{tabular}')
        .replace('BVAR-WN-60', r'\begin{tabular}{@{}c@{}}BVAR \\ ($\lambda_1=0.6$)\end{tabular}')
        .replace('BVAR-WN-80', r'\begin{tabular}{@{}c@{}}BVAR \\ ($\lambda_1=0.8$)\end{tabular}')
        .replace('BVAR-MDD', r'\begin{tabular}{@{}c@{}}BVAR \\ (MDD $\lambda_1$)\end{tabular}')
    )

    
    with open(os.path.join(tables_dir, "Table1_Shrinkage.tex"), "w") as f:
        f.write(latex1)
    print("[SAVED] Table 1: Shrinkage.tex")

    # TABLE 2: Model uncertainty (2x2 Matrix Included)
    cols2 = ['AIC', 'SIC (BIC)', 'HQC', 'OLS-BMA', 'OLS-GEOM-BMA', 'OLS_BMA_AIC', 'OLS_GEOM_BMA_AIC',  'JOINT-BMA', 'GEOM-BMA']
    n2 = len([c for c in cols2 if c in pivot_mse.columns])
    latex2 = styled_mse_latex(
        cols2,
        caption="Relative Mean Squared Error: Hard Selection vs.~Bayesian Model Averaging.",
        label="tab:uncertainty_mse",
        column_format="ll" + "c" * n2,
        position="htbp", position_float="centering", hrules=True,
        multirow_align="t", convert_css=True
    )
    
    latex2 = (latex2
        .replace('\\begin{table}', '\\begin{sidewaystable}')
        .replace('\\end{table}',   '\\end{sidewaystable}')
        .replace('\\begin{tabular}', '\\sbox0{\\begin{tabular}')
        .replace('\\end{tabular}', '\\end{tabular}}\n' + note_text)
        .replace('OLS-BMA', r'\begin{tabular}{@{}c@{}}OLS BMA \\ (BIC)\end{tabular}')
        .replace('OLS-GEOM-BMA', r'\begin{tabular}{@{}c@{}}OLS BMA \\ (Geom. BIC)\end{tabular}')
        .replace('OLS_BMA_AIC', r'\begin{tabular}{@{}c@{}}OLS BMA \\ (AIC)\end{tabular}')
        .replace('OLS_GEOM_BMA_AIC', r'\begin{tabular}{@{}c@{}}OLS BMA \\ (Geom. AIC)\end{tabular}')
        .replace('JOINT-BMA', 'Joint Grid BMA')
        .replace('GEOM-BMA', 'Geom. Grid BMA')
        .replace('SIC (BIC)', 'BIC')
    )
    
    with open(os.path.join(tables_dir, "Table2_Uncertainty.tex"), "w") as f:
        f.write(latex2)
    print("[SAVED] Table 2: Uncertainty.tex")

    # TABLE 3: IC lag orders
    cols3 = ['AIC', 'SIC (BIC)', 'HQC']
    cols3 = [c for c in cols3 if c in pivot_lags.columns]
    latex3 = pivot_lags[cols3].style.format("{:.2f}").to_latex(
        column_format="ll" + "c" * len(cols3),
        position="htbp", position_float="centering", hrules=True,
        caption=r"Average Evaluated Lag Order ($\hat{p}$) selected by Information Criteria.",
        label="tab:ic_lag_orders", multirow_align="t"
    ).replace('SIC (BIC)', 'BIC')
    with open(os.path.join(tables_dir, "Table3_IC_Lags.tex"), "w") as f:
        f.write(latex3)
    print("[SAVED] Table 3: IC_Lags.tex")

    # TABLE 4: BMA posterior lag-weight distribution (All 4 Methods)
    if weights_df is not None:
        weights_renamed = weights_df.rename(columns={'True DGP (p0)': '$p_0$', 'Sample Size (T)': '$T$'})

        bma_exp_lags = df_renamed[df_renamed['Estimator'].isin([
            'OLS BMA (BIC-W)', 'OLS BMA (Geom-W, th=0.5)', 'Joint (p, tau) Grid BMA', 'Geom (p, tau) Grid BMA (th=0.5)'
        ])][['$p_0$', '$T$', 'Estimator', 'Mean Evaluated Lag']].copy()
        
        reverse_est_map = {
            'OLS BMA (BIC-W)': 'OLS-BMA',
            'OLS BMA (Geom-W, th=0.5)': 'OLS-Geom-BMA',
            'Joint (p, tau) Grid BMA': 'Joint-BMA',
            'Geom (p, tau) Grid BMA (th=0.5)': 'Geom-BMA'
        }
        bma_exp_lags['Estimator'] = bma_exp_lags['Estimator'].map(reverse_est_map)

        dist_df = pd.merge(weights_renamed, bma_exp_lags, on=['$p_0$', '$T$', 'Estimator'], how='left')
        
        if not dist_df.empty and 4 in dist_df['$p_0$'].values:
            dist_df_p4 = dist_df[dist_df['$p_0$'] == 4].set_index(['Estimator', '$T$']).drop(columns=['$p_0$'])

            p_cols = [c for c in [f"p={i}" for i in range(1, 13)] if c in dist_df_p4.columns]
            
            if 'Mean Evaluated Lag' in dist_df_p4.columns:
                dist_df_p4 = dist_df_p4[['Mean Evaluated Lag'] + p_cols].rename(columns={'Mean Evaluated Lag': '$E[p]$'})
            else:
                dist_df_p4 = dist_df_p4[p_cols]

            dist_df_str = dist_df_p4.copy()
            if '$E[p]$' in dist_df_str.columns:
                dist_df_str['$E[p]$'] = dist_df_str['$E[p]$'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                
            for col in p_cols:
                dist_df_str[col] = dist_df_str[col].apply(
                    lambda v: "-" if pd.isna(v) or float(v) < 0.001 else f"{float(v)*100:.1f}\\%"
                )

            transposed = dist_df_str.T
            transposed.index.name = 'Statistic / Lag'

            latex4 = transposed.style.to_latex(
                column_format="l|" + "c" * len(transposed.columns),
                position="htbp", position_float="centering", hrules=True,
                caption=r"Posterior Lag-Weight Distribution across all BMA architectures ($p_0 = 4$).",
                label="tab:bma_posterior", multirow_align="t"
            )
            with open(os.path.join(tables_dir, "Table4_BMA_Distribution.tex"), "w") as f:
                f.write(latex4)
            print("[SAVED] Table 4: BMA_Distribution.tex")

    # TABLE 5: BVAR Specifications Relative to AIC
    if 'AIC' in pivot_mse.columns:
        bvar_cols = ['BVAR-WN-20', 'BVAR-WN-40', 'BVAR-WN-60', 'BVAR-WN-80', 'BVAR-MDD']
        bvar_cols_present = [c for c in bvar_cols if c in pivot_mse.columns]
        
        if bvar_cols_present:
            # 1. Calculate relative MSE using AIC as the denominator
            calc_cols = ['AIC'] + bvar_cols_present
            pivot_mse_rel_aic = pivot_mse[calc_cols].div(pivot_mse['AIC'], axis=0)
            
            # 2. Drop the AIC column so it does not appear in the generated table
            pivot_mse_rel_aic = pivot_mse_rel_aic.drop(columns=['AIC'])
            
            # Custom styler for this normalized dataframe
            row_min_aic = pivot_mse_rel_aic.min(axis=1)
            def bold_row_min_aic(row):
                mn = row_min_aic[row.name]
                return ['font-weight: bold' if (pd.notna(v) and pd.notna(mn) and v == mn) else '' 
                        for v in row]

            latex5 = (pivot_mse_rel_aic.style
                    .format("{:.3f}", na_rep="-")
                    .apply(bold_row_min_aic, axis=1)
                    .to_latex(
                        caption=r"Relative Mean Squared Error of Bayesian Shrinkage Specifications vs.~AIC Baseline.",
                        label="tab:bvar_vs_aic",
                        column_format="ll" + "c" * len(bvar_cols_present),
                        position="htbp", position_float="centering", hrules=True,
                        multirow_align="t", convert_css=True
                    ))
            
            # Apply your established LaTeX string replacements (Sideways table formatting removed)
            latex5 = (latex5
                .replace('\\begin{tabular}', '\\sbox0{\\begin{tabular}')
                .replace('\\end{tabular}', '\\end{tabular}}\n' + note_text)
                .replace('BVAR-WN-20', r'\begin{tabular}{@{}c@{}}BVAR \\ ($\lambda_1=0.20$)\end{tabular}')
                .replace('BVAR-WN-40', r'\begin{tabular}{@{}c@{}}BVAR \\ ($\lambda_1=0.40$)\end{tabular}')
                .replace('BVAR-WN-60', r'\begin{tabular}{@{}c@{}}BVAR \\ ($\lambda_1=0.60$)\end{tabular}')
                .replace('BVAR-WN-80', r'\begin{tabular}{@{}c@{}}BVAR \\ ($\lambda_1=0.80$)\end{tabular}')
                .replace('BVAR-MDD', r'\begin{tabular}{@{}c@{}}BVAR \\ (MDD $\lambda_1$)\end{tabular}')
            )
            
            with open(os.path.join(tables_dir, "Table5_BVAR_vs_AIC.tex"), "w") as f:
                f.write(latex5)
            print("[SAVED] Table 5: BVAR_vs_AIC.tex")
    else:
        print("[SKIPPED] Table 5 (AIC column not found in pivot table)")

    # TABLE 6: BMA Specifications Relative to AIC
    if 'AIC' in pivot_mse.columns:
        # Include all BMA models to give Joint Grid BMA proper context
        bma_cols = ['OLS-BMA', 'OLS-GEOM-BMA', 'OLS_BMA_AIC', 'OLS_GEOM_BMA_AIC', 'JOINT-BMA', 'GEOM-BMA']
        bma_cols_present = [c for c in bma_cols if c in pivot_mse.columns]
        
        if bma_cols_present:
            # Calculate relative MSE using AIC as the denominator
            calc_cols = ['AIC'] + bma_cols_present
            pivot_mse_rel_aic_bma = pivot_mse[calc_cols].div(pivot_mse['AIC'], axis=0)
            
            # Drop the AIC column 
            pivot_mse_rel_aic_bma = pivot_mse_rel_aic_bma.drop(columns=['AIC'])
            
            row_min_bma = pivot_mse_rel_aic_bma.min(axis=1)
            def bold_row_min_bma(row):
                mn = row_min_bma[row.name]
                return ['font-weight: bold' if (pd.notna(v) and pd.notna(mn) and v == mn) else '' 
                        for v in row]

            latex6 = (pivot_mse_rel_aic_bma.style
                    .format("{:.3f}", na_rep="-")
                    .apply(bold_row_min_bma, axis=1)
                    .to_latex(
                        caption=r"Relative Mean Squared Error of Bayesian Model Averaging Specifications vs.~AIC Baseline.",
                        label="tab:bma_vs_aic",
                        column_format="ll" + "c" * len(bma_cols_present),
                        position="htbp", position_float="centering", hrules=True,
                        multirow_align="t", convert_css=True
                    ))
            
            latex6 = (latex6
                .replace('\\begin{table}', '\\begin{sidewaystable}')
                .replace('\\end{table}',   '\\end{sidewaystable}')
                .replace('\\begin{tabular}', '\\sbox0{\\begin{tabular}')
                .replace('\\end{tabular}', '\\end{tabular}}\n' + note_text)   
                .replace('OLS-BMA', r'\begin{tabular}{@{}c@{}}OLS BMA \\ (BIC)\end{tabular}')
                .replace('OLS-GEOM-BMA', r'\begin{tabular}{@{}c@{}}OLS BMA \\ (Geom. BIC)\end{tabular}')
                .replace('OLS_BMA_AIC', r'\begin{tabular}{@{}c@{}}OLS BMA \\ (AIC)\end{tabular}')
                .replace('OLS_GEOM_BMA_AIC', r'\begin{tabular}{@{}c@{}}OLS BMA \\ (Geom. AIC)\end{tabular}')
                .replace('JOINT-BMA', 'Joint Grid BMA')
                .replace('GEOM-BMA', 'Geom. Grid BMA')
            )
            
            with open(os.path.join(tables_dir, "Table6_BMA_vs_AIC.tex"), "w") as f:
                f.write(latex6)
            print("[SAVED] Table 6: BMA_vs_AIC.tex")
    else:
        print("[SKIPPED] Table 6 (AIC column not found in pivot table)")

    # -------------------------------------------------------------------
    # 4. FIGURES
    # -------------------------------------------------------------------
    print("\n--- Generating Figures ---")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    
    # Figure 1: Bias-Variance Tradeoff Curve
    # Rebuilt from main df to bypass corrupted tradeoff CSV
    
    # Map the exact estimator strings from the main df to their tau values
    bvar_estimators = {
        'BVAR-WN (tau=0.20, p_max)': 0.20,
        'BVAR-WN (tau=0.40, p_max)': 0.40,
        'BVAR-WN (tau=0.60, p_max)': 0.60,
        'BVAR-WN (tau=0.80, p_max)': 0.80
    }
    
    # Filter main dataframe for p0 = 4 and the relevant BVAR estimators
    tradeoff_rows = df[(df['True DGP (p0)'] == 4) & (df['Sample Size (T)'].isin([240,600])) & (df['Estimator'].isin(bvar_estimators.keys()))]
    
    if not tradeoff_rows.empty:
        # Rebuild the tradeoff data structure
        rebuilt_data = []
        for _, row in tradeoff_rows.iterrows():
            tau_val = bvar_estimators[row['Estimator']]
            t_val = row['Sample Size (T)']
            rel_mse = row['Geom Mean MSE Ratio']
            rebuilt_data.append({'Tau': tau_val, 'T': t_val, 'Rel_MSE': rel_mse})
            
        tradeoff_df_rebuilt = pd.DataFrame(rebuilt_data)
        
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # Plot the main tradeoff curve using the rebuilt data
        sns.lineplot(data=tradeoff_df_rebuilt, x='Tau', y='Rel_MSE', hue='T', marker='o',
                     palette="Set1", linewidth=2.5, ax=ax, errorbar=None)

        # Define baselines: (T_val, linestyle, label, estimator_name)
        baselines = [
            (240, '--', 'AIC Baseline ($T=240$)', 'AIC'),
            (600, '--', 'AIC Baseline ($T=600$)', 'AIC') 
        ]

        for T_val, ls, lbl, estimator in baselines:
            row = df[(df['True DGP (p0)'] == 4) & (df['Sample Size (T)'] == T_val) & (df['Estimator'] == estimator)]
            if len(row):
                color = '#E41A1C' if T_val == 240 else '#377EB8'
                ax.axhline(row['Geom Mean MSE Ratio'].values[0], color=color, linestyle=ls,
                           alpha=0.7, label=lbl)

        for tau in TAU_DISCRETE:
            ax.axvline(tau, color='gray', linestyle=':', linewidth=0.9, alpha=0.5)
            
        ax.axvline(TAU_DISCRETE[0], color='gray', linestyle=':', linewidth=0.9, alpha=0.5,
                   label=r'Estimated $\lambda_1$ grid')

        ax.set_xlabel(r'Shrinkage Parameter ($\lambda_1$)')
        ax.set_ylabel('Relative Mean Squared Error')
        ax.legend(title='Sample Size')
        
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "Fig1_Tradeoff_Curve.pdf"))
        plt.close(fig)
        print("[SAVED] Fig1_Tradeoff_Curve.pdf (Rebuilt from main data)")
    else:
        print("[SKIPPED] Fig1_Tradeoff_Curve (No BVAR-WN data found in main df)")

    # Figure 1: Bias-Variance Tradeoff Curve
    # Rebuilt from main df to bypass corrupted tradeoff CSV
    
    
    # Filter main dataframe for p0 = 4 and the relevant BVAR estimators
    tradeoff_rows = df[(df['True DGP (p0)'] == 10) & (df['Sample Size (T)'].isin([240,600])) & (df['Estimator'].isin(bvar_estimators.keys()))]
    
    if not tradeoff_rows.empty:
        # Rebuild the tradeoff data structure
        rebuilt_data = []
        for _, row in tradeoff_rows.iterrows():
            tau_val = bvar_estimators[row['Estimator']]
            t_val = row['Sample Size (T)']
            rel_mse = row['Geom Mean MSE Ratio']
            rebuilt_data.append({'Tau': tau_val, 'T': t_val, 'Rel_MSE': rel_mse})
            
        tradeoff_df_rebuilt = pd.DataFrame(rebuilt_data)
        
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # Plot the main tradeoff curve using the rebuilt data
        sns.lineplot(data=tradeoff_df_rebuilt, x='Tau', y='Rel_MSE', hue='T', marker='o',
                     palette="Set1", linewidth=2.5, ax=ax, errorbar=None)

        # Define baselines: (T_val, linestyle, label, estimator_name)
        baselines = [
            (240, '--', 'AIC Baseline ($T=240$)', 'AIC'),
            (600, '--', 'AIC Baseline ($T=600$)', 'AIC') 
        ]

        for T_val, ls, lbl, estimator in baselines:
            row = df[(df['True DGP (p0)'] == 4) & (df['Sample Size (T)'] == T_val) & (df['Estimator'] == estimator)]
            if len(row):
                color = '#E41A1C' if T_val == 240 else '#377EB8'
                ax.axhline(row['Geom Mean MSE Ratio'].values[0], color=color, linestyle=ls,
                           alpha=0.7, label=lbl)

        for tau in TAU_DISCRETE:
            ax.axvline(tau, color='gray', linestyle=':', linewidth=0.9, alpha=0.5)
            
        ax.axvline(TAU_DISCRETE[0], color='gray', linestyle=':', linewidth=0.9, alpha=0.5,
                   label=r'Estimated $\lambda_1$ grid')

        ax.set_xlabel(r'Shrinkage Parameter ($\lambda_1$)')
        ax.set_ylabel('Relative Mean Squared Error')
        ax.legend(title='Sample Size')
        
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "Fig1_Tradeoff_Curve_p10.pdf"))
        plt.close(fig)
        print("[SAVED] Fig1_Tradeoff_Curve_p10.pdf (Rebuilt from main data)")
    else:
        print("[SKIPPED] Fig1_Tradeoff_Curve_p10 (No BVAR-WN data found in main df)")
    
    # Figure 1b: Asymptotic Convergence Plot (Performance over T)
    # Isolating the performance of hard selection vs BMA architectures
    
    target_estimators = [
        'AIC',
        'OLS BMA (AIC-W)',
        'Joint (p, tau) Grid BMA'
    ]

    # Filter the main dataframe for the true lag and target estimators
    conv_df = df[(df['True DGP (p0)'] == 4) & (df['Estimator'].isin(target_estimators))].copy()

    if not conv_df.empty:
        # Create cleaner labels for the legend
        label_map = {
            'AIC': 'AIC (Hard Selection)',
            'OLS BMA (AIC-W)': 'OLS BMA (AIC Weights)',
            'Joint (p, tau) Grid BMA': 'Joint Grid BMA'
        }
        conv_df['Model'] = conv_df['Estimator'].map(label_map)

        fig, ax = plt.subplots(figsize=(9, 6))

        # Plotting the convergence lines
        # Using specific markers and distinct colors (Red, Green, Blue) for clarity
        sns.lineplot(data=conv_df, x='Sample Size (T)', y='Geom Mean MSE Ratio',
                     hue='Model', style='Model', markers=['o', 's', 'D'],
                     dashes=['', (2, 2), ''], palette=['#E41A1C', '#4DAF4A', '#377EB8'],
                     linewidth=2.5, markersize=8, ax=ax, errorbar=None)

        ax.set_xlabel('Sample Size ($T$)')
        ax.set_ylabel('Relative Mean Squared Error')

        # Ensure x-axis ticks align perfectly with actual sample sizes in your CSV
        t_values = sorted(conv_df['Sample Size (T)'].unique())
        ax.set_xticks(t_values)

        ax.legend(title='Estimator Architecture', loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

        fig.tight_layout()
        save_name = os.path.join(figures_dir, "Fig1b_Asymptotic_Convergence_p4.pdf")
        fig.savefig(save_name)
        plt.close(fig)
        print(f"[SAVED] Fig1b_Asymptotic_Convergence.pdf")
    else:
        print("[SKIPPED] Fig1b_Asymptotic_Convergence (Required estimators not found in main df)")

    # Figure 1c: Asymptotic Convergence Plot (Performance over T)
    # Isolating the performance of hard selection vs BMA architectures
    
    target_estimators = [
        'AIC',
        'OLS BMA (AIC-W)',
        'Joint (p, tau) Grid BMA'
    ]

    # Filter the main dataframe for the true lag and target estimators
    conv_df = df[(df['True DGP (p0)'] == 10) & (df['Estimator'].isin(target_estimators))].copy()

    if not conv_df.empty:
        # Create cleaner labels for the legend
        label_map = {
            'AIC': 'AIC (Hard Selection)',
            'OLS BMA (AIC-W)': 'OLS BMA (AIC Weights)',
            'Joint (p, tau) Grid BMA': 'Joint Grid BMA'
        }
        conv_df['Model'] = conv_df['Estimator'].map(label_map)

        fig, ax = plt.subplots(figsize=(9, 6))

        # Plotting the convergence lines
        # Using specific markers and distinct colors (Red, Green, Blue) for clarity
        sns.lineplot(data=conv_df, x='Sample Size (T)', y='Geom Mean MSE Ratio',
                     hue='Model', style='Model', markers=['o', 's', 'D'],
                     dashes=['', (2, 2), ''], palette=['#E41A1C', '#4DAF4A', '#377EB8'],
                     linewidth=2.5, markersize=8, ax=ax, errorbar=None)

        ax.set_xlabel('Sample Size ($T$)')
        ax.set_ylabel('Relative Mean Squared Error')

        # Ensure x-axis ticks align perfectly with actual sample sizes in your CSV
        t_values = sorted(conv_df['Sample Size (T)'].unique())
        ax.set_xticks(t_values)

        ax.legend(title='Estimator Architecture', loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

        fig.tight_layout()
        save_name = os.path.join(figures_dir, "Fig1c_Asymptotic_Convergence_p10.pdf")
        fig.savefig(save_name)
        plt.close(fig)
        print(f"[SAVED] Fig1c_Asymptotic_Convergence.pdf")
    else:
        print("[SKIPPED] Fig1c_Asymptotic_Convergence (Required estimators not found in main df)")


    # Figure 2: MDD-selected tau distribution (Evaluated at p_max)
    if os.path.exists(raw_taus_path):
        raw_taus = pd.read_csv(raw_taus_path)
        if 'MDD_Tau' in raw_taus.columns and not raw_taus.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            p0_order = sorted(raw_taus['p0'].unique())
            
            sns.boxplot(data=raw_taus, x='T', y='MDD_Tau', hue='p0', hue_order=p0_order,
                        palette='Blues', showfliers=False, ax=ax,
                        medianprops={'color': 'red', 'linewidth': 2})
            
            # Using palette='Blues' ensures scatter points match the box colors
            sns.stripplot(data=raw_taus, x='T', y='MDD_Tau', hue='p0', hue_order=p0_order,
                          dodge=True, palette='Blues', alpha=0.7, edgecolor='gray', 
                          linewidth=0.5, jitter=True, ax=ax, legend=False)
            
            ax.set_yticks(TAU_DISCRETE)
            ax.set_xlabel('Sample Size ($T$)')
            ax.set_ylabel(r'MDD-Selected Shrinkage Parameter ($\lambda_1$)')
            ax.set_title(r'MDD $\lambda_1$ Selection Distribution across True Lags ($p_{max}$)')
            
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:len(p0_order)], labels[:len(p0_order)], title='True Lag ($p_0$)', loc='best')

            fig.tight_layout()
            fig.savefig(os.path.join(figures_dir, "Fig2_Asymptotic_Relaxation.pdf"))
            plt.close(fig)
            print("[SAVED] Fig2_Asymptotic_Relaxation.pdf")

    # Figure 2b: MDD-selected tau distribution (Evaluated at p0)
    if os.path.exists(raw_taus_p0_path):
        raw_taus_p0 = pd.read_csv(raw_taus_p0_path)
        if 'MDD_Tau_p0' in raw_taus_p0.columns and not raw_taus_p0.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            p0_order = sorted(raw_taus_p0['p0'].unique())
            
            sns.boxplot(data=raw_taus_p0, x='T', y='MDD_Tau_p0', hue='p0', hue_order=p0_order,
                        palette='Greens', showfliers=False, ax=ax,
                        medianprops={'color': 'red', 'linewidth': 2})
            
            # Using palette='Greens' ensures scatter points match the box colors
            sns.stripplot(data=raw_taus_p0, x='T', y='MDD_Tau_p0', hue='p0', hue_order=p0_order,
                          dodge=True, palette='Greens', alpha=0.7, edgecolor='gray', 
                          linewidth=0.5, jitter=True, ax=ax, legend=False)
            
            ax.set_yticks(TAU_DISCRETE)
            ax.set_xlabel('Sample Size ($T$)')
            ax.set_ylabel(r'MDD-Selected Shrinkage Parameter ($\lambda_1$)')
            ax.set_title(r'MDD $\lambda_1$ Selection Distribution Evaluated at True Lag ($p = p_0$)')
            
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:len(p0_order)], labels[:len(p0_order)], title='True Lag ($p_0$)', loc='best')

            fig.tight_layout()
            fig.savefig(os.path.join(figures_dir, "Fig2b_Asymptotic_Relaxation_p0.pdf"))
            plt.close(fig)
            print("[SAVED] Fig2b_Asymptotic_Relaxation_p0.pdf")

    # Figure 3: MDD Surface (Two Random Iterations)
    mdd_df = pd.read_csv(mdd_path)

    if mdd_df is not None and not mdd_df.empty:
        # 2. Filter the dataframe for p0 = 10, T = 600, and the two random iterations.
        filtered_df = mdd_df[
            (mdd_df['T'] == 600) & 
            (mdd_df['p0'] == 10) & 
            (mdd_df['Iter'].isin([27,400]))
        ]

        if not filtered_df.empty:
            fig, ax = plt.subplots(figsize=(9, 6))
        
            # 3. Plot the two iterations as separate lines (hue='iter')
            # We drop the 'errorbar' argument since we are plotting distinct individual paths now.
            sns.lineplot(data=filtered_df, x='Tau', y='MDD', hue='Iter', 
                         palette=['#E41A1C', '#377EB8'], linewidth=2.5, 
                         marker='o', markersize=8, ax=ax)
        
            # 4. Find the peak for each of the two individual iterations to annotate
            colors = ['#E41A1C', '#377EB8']
            # Seaborn assigns palette colors based on sorted hue values
            sorted_iters = sorted(filtered_df['Iter'].unique())

            for iteration, color in zip(sorted_iters, colors):
                iter_df = filtered_df[filtered_df['Iter'] == iteration]
                if not iter_df.empty:
                    # Find the index of the maximum MDD, then extract the corresponding Tau
                    max_tau = iter_df.loc[iter_df['MDD'].idxmax(), 'Tau']
                
                    ax.axvline(max_tau, color=color, linestyle='--', alpha=0.6,
                               label=fr'Optimal $\lambda_1 \simeq {max_tau:.2f}$ (Iter {iteration})')

            ax.set_xlabel(r'Candidate Shrinkage Parameter ($\lambda_1$)')
            ax.set_ylabel('Marginal Data Density (Log Likelihood)')
        
            # Update title to reflect the fixed T and p0 parameters
            ax.set_title(r'Objective Function Surface ($T=600$, $p_0=10$)')
        
            # Clean up legend to reflect iterations instead of T
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(title='Iteration', loc='best')
        
            fig.tight_layout()
            fig.savefig(os.path.join(figures_dir, "Fig3_MDD_Surface.pdf"))
            plt.close(fig)
            print("[SAVED] Fig3_MDD_Surface.pdf")
        else:
            print("[SKIPPED] Fig3_MDD_Surface (Filtered data is empty. Check column names 'p0' and 'iter')")
    else:
        print("[SKIPPED] Fig3_MDD_Surface (No data found)") 


    # Figure 5: BMA Posterior Weight Heatmaps (All 6 Included)
    if weights_df is not None:
        heatmap_configs = [
            ('OLS-BMA',      'OLS',      r'OLS BMA (BIC-W)',             'Fig5a'),
            ('OLS-Geom-BMA', 'OLS_Geom', r'OLS BMA (Geom-BIC-W, $\theta=0.5$)', 'Fig5b'),
            ('OLS-BMA (AIC-W)',      'OLS_AIC',      r'OLS BMA (AIC-W)',             'Fig5c'),
            ('OLS-Geom-BMA (AIC-W)', 'OLS_Geom_AIC', r'OLS BMA (Geom-AIC-W, $\theta=0.5$)', 'Fig5d'),
            ('Joint-BMA',    'Joint',    r'Joint BMA ($p, \lambda_1$)',       'Fig5e'),
            ('Geom-BMA',     'Geom',     r'Geom BMA ($\theta=0.5$)',     'Fig5f'),
        ]
        
        for est_name, file_suffix, title_str, fig_num in heatmap_configs:
            heat_df = weights_df[
                (weights_df['Estimator'] == est_name) &
                (weights_df['True DGP (p0)'] == 4)
            ].copy().set_index('Sample Size (T)')
            
            if heat_df.empty:
                print(f"[SKIPPED] {fig_num} (No data for {est_name} in weights CSV)")
                continue

            p_cols = [c for c in [f"p={i}" for i in range(1, 13)] if c in heat_df.columns]
            heat_df = heat_df[p_cols].T
            heat_df.index.name = 'Candidate Lag'

            fig, ax = plt.subplots(figsize=(8, 7))
            sns.heatmap(heat_df, annot=True, fmt=".1%", cmap="YlGnBu",
                        cbar_kws={'label': 'Posterior Probability Mass'},
                        linewidths=1, linecolor='white', ax=ax)

            true_lag_idx = 3  # p=4 is the 4th row (0-indexed = 3)
            ax.add_patch(plt.Rectangle((0, true_lag_idx), len(heat_df.columns), 1,
                                       fill=False, edgecolor='red', lw=3, clip_on=False))

            ax.set_title(f'Posterior Weight Evolution: {title_str} ($p_0=4$)')
            ax.set_xlabel('Sample Size ($T$)')
            ax.set_ylabel('Candidate Lag Order ($p$)')
            fig.tight_layout()
            
            save_name = f"{fig_num}_BMA_Heatmap_{file_suffix}.pdf"
            fig.savefig(os.path.join(figures_dir, save_name))
            plt.close(fig)
            print(f"[SAVED] {save_name}")

    # Figure 9, tau dependent on T
    # Assuming tau_path and figures_dir are defined earlier in your script
    target_T = 600
    target_p0 = 4  # <-- Define your target p0 here

    if os.path.exists(tau_path):
        df = pd.read_csv(tau_path)
    
        # 1. Filter for a specific Sample Size AND a specific True Lag (p0)
        df_filtered = df[(df['T'] == target_T) & (df['p0'] == target_p0)]
    
        # 2. Explicitly aggregate the Monte Carlo iterations
        # This takes the mean of 'Expected_Tau' for each combination of 'p' and 'Estimator'
        df_avg = df_filtered.groupby(['p', 'Estimator'])['Expected_Tau'].mean().reset_index()
    
        fig, ax = plt.subplots(figsize=(8, 6))
    
        # 3. Plot the newly averaged data
        # We can drop errorbar=None since df_avg now has exactly one row per x/hue combination
        sns.lineplot(data=df_avg, x='p', y='Expected_Tau', 
                 hue='Estimator', palette=['#1f77b4', '#ff7f0e'], 
                 marker='o', linewidth=2.5, ax=ax)
    
        # Updated title to reflect both T and p0
        ax.set_title(fr'Optimal Shrinkage vs. Lag Order ($T={target_T}, p_0={target_p0}$)')
        ax.set_xlabel('Candidate Lag Order ($p$)')
        ax.set_ylabel(r'Expected Shrinkage ($\mathbb{E}[\lambda_1 \mid p]$)')
    
        # Set x-ticks dynamically based on the aggregated data
        ax.set_xticks(range(1, int(df_avg['p'].max()) + 1))
        ax.grid(True, linestyle='--', alpha=0.6)
    
        fig.tight_layout()
        save_name = os.path.join(figures_dir, f"Fig9_Tau_Comparison_T{target_T}_p0{target_p0}.pdf")
        fig.savefig(save_name)
        plt.close(fig)
        print(f"[SAVED] {save_name}")

    print("\n" + "="*80)
    print(" VISUALIZATION PROCESS COMPLETE!")
    print(f" Tables verified at:  {tables_dir}")
    print(f" Figures verified at: {figures_dir}")
    print("="*80)

    # Figure 9b, tau dependent on T
    # Assuming tau_path and figures_dir are defined earlier in your script
    target_T = 600
    target_p0 = 10  # <-- Define your target p0 here

    if os.path.exists(tau_path):
        df = pd.read_csv(tau_path)
    
        # 1. Filter for a specific Sample Size AND a specific True Lag (p0)
        df_filtered = df[(df['T'] == target_T) & (df['p0'] == target_p0)]
    
        # 2. Explicitly aggregate the Monte Carlo iterations
        # This takes the mean of 'Expected_Tau' for each combination of 'p' and 'Estimator'
        df_avg = df_filtered.groupby(['p', 'Estimator'])['Expected_Tau'].mean().reset_index()
    
        fig, ax = plt.subplots(figsize=(8, 6))
    
        # 3. Plot the newly averaged data
        # We can drop errorbar=None since df_avg now has exactly one row per x/hue combination
        sns.lineplot(data=df_avg, x='p', y='Expected_Tau', 
                 hue='Estimator', palette=['#1f77b4', '#ff7f0e'], 
                 marker='o', linewidth=2.5, ax=ax)
    
        # Updated title to reflect both T and p0
        ax.set_title(fr'Optimal Shrinkage vs. Lag Order ($T={target_T}, p_0={target_p0}$)')
        ax.set_xlabel('Candidate Lag Order ($p$)')
        ax.set_ylabel(r'Expected Shrinkage ($\mathbb{E}[\lambda_1 \mid p]$)')
    
        # Set x-ticks dynamically based on the aggregated data
        ax.set_xticks(range(1, int(df_avg['p'].max()) + 1))
        ax.grid(True, linestyle='--', alpha=0.6)
    
        fig.tight_layout()
        save_name = os.path.join(figures_dir, f"Fig9_Tau_Comparison_T{target_T}_p0{target_p0}.pdf")
        fig.savefig(save_name)
        plt.close(fig)
        print(f"[SAVED] {save_name}")

    # Figure 5: BMA Posterior Weight Heatmaps (All 6 Included)
    if weights_df is not None:
        heatmap_configs = [
            ('OLS-BMA',      'OLS',      r'OLS BMA (BIC-W)',             'Fig5a'),
            ('OLS-Geom-BMA', 'OLS_Geom', r'OLS BMA (Geom-BIC-W, $\theta=0.5$)', 'Fig5b'),
            ('OLS-BMA (AIC-W)',      'OLS_AIC',      r'OLS BMA (AIC-W)',             'Fig5c'),
            ('OLS-Geom-BMA (AIC-W)', 'OLS_Geom_AIC', r'OLS BMA (Geom-AIC-W, $\theta=0.5$)', 'Fig5d'),
            ('Joint-BMA',    'Joint',    r'Joint BMA ($p, \lambda_1$)',       'Fig5e'),
            ('Geom-BMA',     'Geom',     r'Geom BMA ($\theta=0.5$)',     'Fig5f'),
        ]
        
        for est_name, file_suffix, title_str, fig_num in heatmap_configs:
            heat_df = weights_df[
                (weights_df['Estimator'] == est_name) &
                (weights_df['True DGP (p0)'] == 10)
            ].copy().set_index('Sample Size (T)')
            
            if heat_df.empty:
                print(f"[SKIPPED] {fig_num} (No data for {est_name} in weights CSV)")
                continue

            p_cols = [c for c in [f"p={i}" for i in range(1, 13)] if c in heat_df.columns]
            heat_df = heat_df[p_cols].T
            heat_df.index.name = 'Candidate Lag'

            fig, ax = plt.subplots(figsize=(8, 7))
            sns.heatmap(heat_df, annot=True, fmt=".1%", cmap="YlGnBu",
                        cbar_kws={'label': 'Posterior Probability Mass'},
                        linewidths=1, linecolor='white', ax=ax)

            true_lag_idx = 3  # p=4 is the 4th row (0-indexed = 3)
            ax.add_patch(plt.Rectangle((0, true_lag_idx), len(heat_df.columns), 1,
                                       fill=False, edgecolor='red', lw=3, clip_on=False))

            ax.set_title(f'Posterior Weight Evolution: {title_str} ($p_0=4$)')
            ax.set_xlabel('Sample Size ($T$)')
            ax.set_ylabel('Candidate Lag Order ($p$)')
            fig.tight_layout()
            
            save_name = f"{fig_num}_BMA_Heatmap_{file_suffix}_p10.pdf"
            fig.savefig(os.path.join(figures_dir, save_name))
            plt.close(fig)
            print(f"[SAVED] {save_name}")


    print("\n" + "="*80)
    print(" VISUALIZATION PROCESS COMPLETE!")
    print(f" Tables verified at:  {tables_dir}")
    print(f" Figures verified at: {figures_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
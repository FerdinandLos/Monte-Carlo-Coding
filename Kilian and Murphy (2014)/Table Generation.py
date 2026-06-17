import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # -------------------------------------------------------------------
    # 1. ROBUST PATH RESOLUTION
    # -------------------------------------------------------------------
    current_script_location = os.path.dirname(os.path.abspath(__file__))
    root_project_folder = os.path.dirname(current_script_location)
    
    KM_FOLDER = 'Kilian and Murphy (2014)'
    km_base = os.path.join(root_project_folder, KM_FOLDER)
    results_dir = os.path.join(km_base, 'Results')

    # Ensure output directories exist
    tables_dir = os.path.join(root_project_folder, "Writing", "Tables")
    figures_dir = os.path.join(root_project_folder, "Writing", "Figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # 2. LOAD ALL CSV DATAFRAMES
    # -------------------------------------------------------------------
    main_path = os.path.join(results_dir, "Master_Final_SVAR_Comparison_iters100_draws50.csv")
    weights_path = os.path.join(results_dir, "Master_BMA_Weights_iters100.csv")
    raw_taus_path = os.path.join(results_dir, "Master_Raw_Opt_Tau.csv")
    tradeoff_path = os.path.join(results_dir, "Master_Tradeoff_Curve.csv")
    mdd_path = os.path.join(results_dir, "Master_MDD_Surface.csv")
    iter_mse_path = os.path.join(results_dir, "Master_Iteration_MSEs.csv")
    
    if not os.path.exists(main_path):
        print(f"\n[ERROR] Main results CSV not found in:\n{results_dir}")
        return
            
    df = pd.read_csv(main_path)
    weights_df = pd.read_csv(weights_path) if os.path.exists(weights_path) else None

    # -------------------------------------------------------------------
    # 3. GENERATE THE 4 TABLES
    # -------------------------------------------------------------------
    df_renamed = df.rename(columns={'True DGP (p0)': '$p_0$', 'Sample Size (T)': '$T$'})

    estimator_mapping = {
        'BVAR-WN (Tight tau=0.05)': 'BVAR_TIGHT_05',
        'BVAR-WN (Std tau=0.20)': 'BVAR_STD_20',
        'BVAR-WN (Loose tau=0.50)': 'BVAR_LOOSE_50',
        'BVAR-RW (Std tau=0.20)': 'BVAR_RW_20',
        'Hybrid-BVAR (OLS p, Fix tau)': 'BVAR_BIC',
        'Hybrid-BMA (OLS W, Fix tau)': 'BVAR_BMA',
        'SOTA-BVAR (MDD p, Opt tau)': 'SOTA_BVAR',
        'SOTA-BMA (MDD W, Opt tau)': 'SOTA_BMA'
    }
    df_renamed['Estimator_Mapped'] = df_renamed['Estimator'].replace(estimator_mapping)

    # Pivot MSE so we can format it. For Tables 1 & 2 we want T as the outer
    # (spanning) row index and p0 as the inner one, ordered by T.
    pivot_mse = df_renamed.pivot(index=['$T$', '$p_0$'], columns='Estimator_Mapped', values='Geom Mean MSE Ratio').sort_index()
    pivot_lags = df_renamed.pivot(index=['$p_0$', '$T$'], columns='Estimator', values='Mean Evaluated Lag')

    # Row-min over the FULL estimator set (matches original bolding logic, which
    # selects the winner across all estimators, not just the table's subset).
    row_min = pivot_mse.min(axis=1)

    def bold_full_row_min(row):
        # Bold cells equal to the full-row minimum; '' otherwise. NaNs left blank.
        return ['font-weight: bold' if (pd.notna(v) and v == row_min[row.name]) else ''
                for v in row]

    # Strip axis names so the Styler header row doesn't print 'Estimator_Mapped'.
    pivot_mse = pivot_mse.rename_axis(columns=None)

    def styled_mse_latex(cols, **to_latex_kwargs):
        return (pivot_mse[cols].style
                .format("{:.3f}", na_rep="-")
                .apply(bold_full_row_min, axis=1)
                .to_latex(**to_latex_kwargs))

    # Table 1: Shrinkage Strategy
    cols_paper1 = ['AIC', 'AICc', 'SIC (BIC)', 'HQC', 'BVAR_TIGHT_05', 'BVAR_STD_20', 'BVAR_LOOSE_50', 'BVAR_RW_20', 'SOTA_BVAR', 'SOTA_BMA']
    cols_paper1 = [c for c in cols_paper1 if c in pivot_mse.columns]
    
    latex_paper1 = styled_mse_latex(cols_paper1,
        column_format="ll" + "c" * len(cols_paper1), position="htbp", position_float="centering", hrules=True,
        caption="Relative Mean Squared Error: Hard Selection vs. Bayesian Shrinkage.", label="tab:shrinkage_mse", multirow_align="t",
        convert_css=True
    ).replace('\\begin{table}', '\\begin{sidewaystable}').replace('\\end{table}', '\\end{sidewaystable}')
    
    latex_paper1 = latex_paper1.replace('BVAR_TIGHT_05', r'\begin{tabular}{@{}c@{}}BVAR-WN \\ ($\tau=0.05$)\end{tabular}')\
        .replace('BVAR_STD_20', r'\begin{tabular}{@{}c@{}}BVAR-WN \\ ($\tau=0.20$)\end{tabular}')\
        .replace('BVAR_LOOSE_50', r'\begin{tabular}{@{}c@{}}BVAR-WN \\ ($\tau=0.50$)\end{tabular}')\
        .replace('BVAR_RW_20', r'\begin{tabular}{@{}c@{}}BVAR-RW \\ ($\tau=0.20$)\end{tabular}')\
        .replace('SOTA_BVAR', r'\begin{tabular}{@{}c@{}}SOTA-BVAR (WN) \\ (Opt $\tau$)\end{tabular}')\
        .replace('SOTA_BMA', r'\begin{tabular}{@{}c@{}}SOTA-BMA (WN) \\ (Opt $\tau$)\end{tabular}')

    with open(os.path.join(tables_dir, "Table_Paper1_Shrinkage.tex"), "w") as f: f.write(latex_paper1)

    # Table 2: Uncertainty Strategy
    cols_paper2 = ['AIC', 'AICc', 'SIC (BIC)', 'HQC', 'OLS BMA (BIC-W)', 'BVAR_BIC', 'BVAR_BMA', 'SOTA_BMA']
    cols_paper2 = [c for c in cols_paper2 if c in pivot_mse.columns]

    latex_paper2 = styled_mse_latex(cols_paper2,
        column_format="ll" + "c" * len(cols_paper2), position="htbp", position_float="centering", hrules=True,
        caption="Relative Mean Squared Error: Hard Selection vs. Bayesian Model Averaging.", label="tab:uncertainty_mse", multirow_align="t",
        convert_css=True
    ).replace('\\begin{table}', '\\begin{sidewaystable}').replace('\\end{table}', '\\end{sidewaystable}')
    
    latex_paper2 = latex_paper2.replace('BVAR_BIC', r'\begin{tabular}{@{}c@{}}Hybrid-BVAR (WN) \\ (Fixed $\tau$)\end{tabular}')\
        .replace('BVAR_BMA', r'\begin{tabular}{@{}c@{}}Hybrid-BMA (WN) \\ (Fixed $\tau$)\end{tabular}')\
        .replace('SOTA_BMA', r'\begin{tabular}{@{}c@{}}SOTA-BMA (WN) \\ (Opt $\tau$)\end{tabular}')

    with open(os.path.join(tables_dir, "Table_Paper2_Uncertainty.tex"), "w") as f: f.write(latex_paper2)

    # Table 3: IC Lag Orders
    cols_paper3 = ['AIC', 'AICc', 'SIC (BIC)', 'HQC']
    cols_paper3 = [c for c in cols_paper3 if c in pivot_lags.columns]
    
    latex_paper3 = pivot_lags[cols_paper3].style.format("{:.2f}").to_latex(
        column_format="ll" + "c" * len(cols_paper3), position="htbp", position_float="centering", hrules=True,
        caption=r"Average Evaluated Lag Order ($\hat{p}$) selected by standard Information Criteria.", label="tab:ic_lag_orders", multirow_align="t")
    
    with open(os.path.join(tables_dir, "Table_Paper3_IC_Lags.tex"), "w") as f: f.write(latex_paper3)

    # Table 4: BMA Weights (Transposed for standard portrait layout)
    if weights_df is not None:
        weights_renamed = weights_df.rename(columns={'True DGP (p0)': '$p_0$', 'Sample Size (T)': '$T$'})
        bma_expected_lags = df_renamed[df_renamed['Estimator'].isin(['Hybrid-BMA (OLS W, Fix tau)', 'SOTA-BMA (MDD W, Opt tau)'])].copy()
        bma_expected_lags['Estimator'] = bma_expected_lags['Estimator'].map({'Hybrid-BMA (OLS W, Fix tau)': 'Hybrid-BMA (OLS W)', 'SOTA-BMA (MDD W, Opt tau)': 'SOTA-BMA (MDD W)'})
        bma_expected_lags = bma_expected_lags[['$p_0$', '$T$', 'Estimator', 'Mean Evaluated Lag']].rename(columns={'Mean Evaluated Lag': '$E[p]$'})
        
        dist_df = pd.merge(weights_renamed, bma_expected_lags, on=['$p_0$', '$T$', 'Estimator'], how='left')
        dist_df_p4 = dist_df[dist_df['$p_0$'] == 4].set_index(['Estimator', '$T$']).drop(columns=['$p_0$'])
        
        p_cols = [c for c in [f"p={i}" for i in range(1, 13)] if c in dist_df_p4.columns]
        dist_df_p4 = dist_df_p4[['$E[p]$'] + p_cols]

        dist_df_p4_str = dist_df_p4.copy()
        dist_df_p4_str['$E[p]$'] = dist_df_p4_str['$E[p]$'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        
        def format_weight(val):
            # FIXED: We multiply by 100 and append \% manually to avoid f-string format crashes
            return "-" if pd.isna(val) or float(val) < 0.001 else f"{float(val) * 100:.1f}\\%"
            
        for col in p_cols:
            dist_df_p4_str[col] = dist_df_p4_str[col].apply(format_weight)
            
        transposed_df = dist_df_p4_str.T
        transposed_df.index.name = 'Statistic / Lag'

        latex_paper4 = (transposed_df.style
            .to_latex(column_format="l|" + "c" * len(transposed_df.columns), 
                      position="htbp", position_float="centering", hrules=True,
                      caption="Posterior Lag Weight Distribution for BMA ($p_0 = 4$).", 
                      label="tab:bma_posterior", multirow_align="t")
        )
        
        latex_paper4 = latex_paper4.replace(
            'Hybrid-BMA (OLS W)', r'\begin{tabular}{@{}c@{}}Hybrid-BMA \\ (OLS W)\end{tabular}'
        ).replace(
            'SOTA-BMA (MDD W)', r'\begin{tabular}{@{}c@{}}SOTA-BMA \\ (MDD W)\end{tabular}'
        )
        
        with open(os.path.join(tables_dir, "Table_Paper4_BMA_Distribution.tex"), "w") as f: f.write(latex_paper4)

    # -------------------------------------------------------------------
    # 4. GENERATE THE 5 PLOTS
    # -------------------------------------------------------------------
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Graph 1: Bias-Variance Tradeoff
    if os.path.exists(tradeoff_path):
        tradeoff_df = pd.read_csv(tradeoff_path)
        plt.figure(figsize=(9, 6))
        sns.lineplot(data=tradeoff_df, x='Tau', y='Rel_MSE', hue='T', marker='o', palette="Set1", linewidth=2.5)
        
        sic_96 = df[(df['True DGP (p0)'] == 4) & (df['Sample Size (T)'] == 96) & (df['Estimator'] == 'SIC (BIC)')]['Geom Mean MSE Ratio'].values[0]
        sic_480 = df[(df['True DGP (p0)'] == 4) & (df['Sample Size (T)'] == 480) & (df['Estimator'] == 'SIC (BIC)')]['Geom Mean MSE Ratio'].values[0]
        
        plt.axhline(sic_96, color='#E41A1C', linestyle='--', alpha=0.7, label='SIC Baseline (T=96)')
        plt.axhline(sic_480, color='#377EB8', linestyle='--', alpha=0.7, label='SIC Baseline (T=480)')
        
        plt.xlabel(r'Shrinkage Parameter ($\tau$)')
        plt.ylabel('Relative Mean Squared Error')
        plt.title(r'Bias-Variance Tradeoff Curve ($p_0 = 4$)')
        plt.legend(title='Sample Size')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "Fig1_Tradeoff_Curve.pdf"))
        plt.close()

    # Graph 2: Asymptotic Relaxation Boxplot
    if os.path.exists(raw_taus_path):
        raw_taus = pd.read_csv(raw_taus_path)
        raw_taus_p4 = raw_taus[raw_taus['p0'] == 4]
        
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=raw_taus_p4, x='T', y='Opt_Tau', hue='T', palette='Blues', 
                    legend=False, showfliers=False)
        sns.stripplot(data=raw_taus_p4, x='T', y='Opt_Tau', color='black', alpha=0.3, jitter=True)
        
        plt.xlabel('Sample Size ($T$)')
        plt.ylabel(r'Optimized Shrinkage Parameter ($\tau$)')
        plt.title(r'Asymptotic Relaxation of the Minnesota Prior ($p_0 = 4$)')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "Fig2_Asymptotic_Relaxation.pdf"))
        plt.close()

    # Graph 3: MDD Surface
    if os.path.exists(mdd_path):
        mdd_df = pd.read_csv(mdd_path)
        max_idx = mdd_df['MDD'].idxmax()
        max_tau = mdd_df['Tau'][max_idx]
        max_mdd = mdd_df['MDD'][max_idx]

        plt.figure(figsize=(8, 6))
        sns.lineplot(data=mdd_df, x='Tau', y='MDD', color='purple', linewidth=2.5)
        # FIXED: Escaped the \simeq with double backslashes
        plt.axvline(max_tau, color='black', linestyle='--', alpha=0.6, label=f'Optimal $\\tau \\simeq {max_tau:.2f}$')
        plt.plot(max_tau, max_mdd, 'ko', markersize=8)
        
        plt.xlabel(r'Candidate Shrinkage Parameter ($\tau$)')
        plt.ylabel('Marginal Data Density (Log Likelihood)')
        plt.title('Objective Function Surface ($T = 480$)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "Fig3_MDD_Surface.pdf"))
        plt.close()

    # Graph 4: Risk Hedging Scatter Plot (BMA vs BIC)
    if os.path.exists(iter_mse_path):
        iter_df = pd.read_csv(iter_mse_path)
        scatter_df = iter_df[(iter_df['p0'] == 4) & (iter_df['T'] == 96)]
        
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=scatter_df, x='BMA_Rel_MSE', y='BIC_Rel_MSE', 
                        alpha=0.6, color='#2CA02C', edgecolor='black', s=60)
        
        max_val = max(scatter_df['BMA_Rel_MSE'].max(), scatter_df['BIC_Rel_MSE'].max())
        plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Equal Performance ($y=x$)')
        plt.fill_between([0, max_val], [0, max_val], max_val * 1.5, color='green', alpha=0.05, label='BMA Outperforms')
        
        plt.xlim(left=0.5) 
        plt.ylim(bottom=0.5)
        plt.xlabel('SOTA-BMA Relative MSE')
        plt.ylabel('Hard Selection (BIC) Relative MSE')
        plt.title('Catastrophe Avoidance: BMA vs. Hard Selection ($p_0=4, T=96$)')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "Fig4_Risk_Hedging_Scatter.pdf"))
        plt.close()

    # Graph 5: BMA Posterior Evolution Heatmap
    if weights_df is not None:
        heat_df = weights_df[(weights_df['Estimator'] == 'SOTA-BMA (MDD W)') & (weights_df['True DGP (p0)'] == 4)].copy()
        heat_df = heat_df.set_index('Sample Size (T)')
        
        p_cols = [c for c in [f"p={i}" for i in range(1, 13)] if c in heat_df.columns]
        heat_df = heat_df[p_cols].T
        heat_df.index.name = 'Candidate Lag'
        
        plt.figure(figsize=(8, 7))
        ax = sns.heatmap(heat_df, annot=True, fmt=".1%", cmap="YlGnBu", 
                         cbar_kws={'label': 'Posterior Probability Mass'},
                         linewidths=1, linecolor='white')
        
        true_lag_idx = 3 # p=4 is index 3
        ax.add_patch(plt.Rectangle((0, true_lag_idx), len(heat_df.columns), 1, 
                                   fill=False, edgecolor='red', lw=3, clip_on=False))
        
        plt.title('Posterior Weight Evolution of SOTA-BMA ($p_0=4$)')
        plt.xlabel('Sample Size ($T$)')
        plt.ylabel('Candidate Lag Order ($p$)')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "Fig5_BMA_Heatmap.pdf"))
        plt.close()

    print("\n" + "="*80)
    print(" SUCCESS: Tables and Figures generated!")
    print(f" Tables saved to:  {tables_dir}")
    print(f" Figures saved to: {figures_dir}")
    print("="*80)

if __name__ == '__main__':
    main()

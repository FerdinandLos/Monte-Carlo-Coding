import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

TAU_DISCRETE = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]

def main():
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
    main_path      = os.path.join(results_dir, "Master_Final_SVAR_Comparison_iters100_draws50.csv")
    weights_path   = os.path.join(results_dir, "Master_BMA_Weights_iters100_draws50.csv")
    raw_taus_path  = os.path.join(results_dir, "Master_Raw_MDD_Tau_iters100_draws50.csv")
    tradeoff_path  = os.path.join(results_dir, "Master_Tradeoff_Curve_iters100_draws50.csv")
    mdd_path       = os.path.join(results_dir, "Master_MDD_Surface_iters100_draws50.csv")
    iter_mse_path  = os.path.join(results_dir, "Master_Iteration_MSEs_iters100_draws50.csv")

    if not os.path.exists(main_path):
        print(f"\n[ERROR] Main results CSV not found in:\n{results_dir}")
        return

    df = pd.read_csv(main_path)
    weights_df = pd.read_csv(weights_path) if os.path.exists(weights_path) else None

    # -------------------------------------------------------------------
    # 3. PREPARE PIVOT TABLES
    # -------------------------------------------------------------------
    df_renamed = df.rename(columns={'True DGP (p0)': '$p_0$', 'Sample Size (T)': '$T$'})

    # Map verbose estimator names to short tokens used as pivot column keys
    estimator_mapping = {
        'BVAR-WN (tau=0.05, p_max)': 'BVAR_WN_05',
        'BVAR-WN (tau=0.20, p_max)': 'BVAR_WN_20',
        'BVAR-WN (tau=0.35, p_max)': 'BVAR_WN_35',
        'BVAR-WN (tau=0.50, p_max)': 'BVAR_WN_50',
        'BVAR-WN (tau=0.65, p_max)': 'BVAR_WN_65',
        'BVAR-WN (tau=0.80, p_max)': 'BVAR_WN_80',
        'BVAR-WN (tau=0.95, p_max)': 'BVAR_WN_95',
        'BVAR-RW (tau=0.20, p_max)': 'BVAR_RW_20',
        'BVAR-RW (tau=0.50, p_max)': 'BVAR_RW_50',
        'BMA BVAR-WN (BIC-W, tau=0.20)': 'BMA_BVAR_20',
        'BMA BVAR-WN (BIC-W, tau=0.50)': 'BMA_BVAR_50',
        'BVAR-WN (MDD tau, p_max)': 'BVAR_MDD',
    }
    df_renamed['Estimator_Mapped'] = df_renamed['Estimator'].replace(estimator_mapping)

    pivot_mse = (df_renamed
                 .pivot(index=['$T$', '$p_0$'], columns='Estimator_Mapped', values='Geom Mean MSE Ratio')
                 .sort_index()
                 .rename_axis(columns=None))

    pivot_lags = df_renamed.pivot(index=['$p_0$', '$T$'], columns='Estimator', values='Mean Evaluated Lag')

    # -------------------------------------------------------------------
    # Bold-min helper: computes the minimum per-table over the displayed
    # columns only, so the bold winner is always visible in the table.
    # -------------------------------------------------------------------
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

    # -------------------------------------------------------------------
    # TABLE 1: Shrinkage strategy — IC + all BVAR fixed-tau + MDD-tau
    # (no BMA models)
    # -------------------------------------------------------------------
    cols1 = ['AIC', 'SIC (BIC)', 'HQC',
             'BVAR_WN_05', 'BVAR_WN_20', 'BVAR_WN_35', 'BVAR_WN_50',
             'BVAR_WN_65', 'BVAR_WN_80', 'BVAR_WN_95',
             'BVAR_RW_20', 'BVAR_RW_50', 'BVAR_MDD']

    n1 = len([c for c in cols1 if c in pivot_mse.columns])
    latex1 = styled_mse_latex(
        cols1,
        caption="Relative Mean Squared Error: Hard Selection vs.~Bayesian Shrinkage (all fixed-$\\tau$ BVAR-WN and BVAR-RW, plus MDD-selected $\\tau$).",
        label="tab:shrinkage_mse",
        column_format="ll" + "c" * n1,
        position="htbp", position_float="centering", hrules=True,
        multirow_align="t", convert_css=True
    )
    latex1 = (latex1
        .replace('\\begin{table}', '\\begin{sidewaystable}')
        .replace('\\end{table}',   '\\end{sidewaystable}')
        .replace('BVAR_WN_05', r'\begin{tabular}{@{}c@{}}BVAR-WN \\ ($\tau=0.05$)\end{tabular}')
        .replace('BVAR_WN_20', r'\begin{tabular}{@{}c@{}}BVAR-WN \\ ($\tau=0.20$)\end{tabular}')
        .replace('BVAR_WN_35', r'\begin{tabular}{@{}c@{}}BVAR-WN \\ ($\tau=0.35$)\end{tabular}')
        .replace('BVAR_WN_50', r'\begin{tabular}{@{}c@{}}BVAR-WN \\ ($\tau=0.50$)\end{tabular}')
        .replace('BVAR_WN_65', r'\begin{tabular}{@{}c@{}}BVAR-WN \\ ($\tau=0.65$)\end{tabular}')
        .replace('BVAR_WN_80', r'\begin{tabular}{@{}c@{}}BVAR-WN \\ ($\tau=0.80$)\end{tabular}')
        .replace('BVAR_WN_95', r'\begin{tabular}{@{}c@{}}BVAR-WN \\ ($\tau=0.95$)\end{tabular}')
        .replace('BVAR_RW_20', r'\begin{tabular}{@{}c@{}}BVAR-RW \\ ($\tau=0.20$)\end{tabular}')
        .replace('BVAR_RW_50', r'\begin{tabular}{@{}c@{}}BVAR-RW \\ ($\tau=0.50$)\end{tabular}')
        .replace('BVAR_MDD',   r'\begin{tabular}{@{}c@{}}BVAR-WN \\ (MDD $\tau$)\end{tabular}')
        .replace('SIC (BIC)',  'BIC')
    )
    with open(os.path.join(tables_dir, "Table1_Shrinkage.tex"), "w") as f:
        f.write(latex1)

    # -------------------------------------------------------------------
    # TABLE 2: Model uncertainty — IC + OLS BMA + both BMA BVAR-WN
    # -------------------------------------------------------------------
    cols2 = ['AIC', 'SIC (BIC)', 'HQC',
             'OLS BMA (BIC-W)', 'BMA_BVAR_20', 'BMA_BVAR_50']

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
        .replace('OLS BMA (BIC-W)', r'\begin{tabular}{@{}c@{}}OLS BMA \\ (BIC-W)\end{tabular}')
        .replace('BMA_BVAR_20', r'\begin{tabular}{@{}c@{}}BMA BVAR-WN \\ (BIC-W, $\tau=0.20$)\end{tabular}')
        .replace('BMA_BVAR_50', r'\begin{tabular}{@{}c@{}}BMA BVAR-WN \\ (BIC-W, $\tau=0.50$)\end{tabular}')
        .replace('SIC (BIC)', 'BIC')
    )
    with open(os.path.join(tables_dir, "Table2_Uncertainty.tex"), "w") as f:
        f.write(latex2)

    # -------------------------------------------------------------------
    # TABLE 3: IC lag orders — AIC, BIC, HQC only (no AICc)
    # -------------------------------------------------------------------
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

    # -------------------------------------------------------------------
    # TABLE 4: BMA posterior lag-weight distribution
    # Both OLS BMA and BMA BVAR-WN share the same BIC lag weights, so we
    # show the single "Hybrid-BMA (OLS W)" weight distribution from the CSV.
    # -------------------------------------------------------------------
    if weights_df is not None:
        weights_renamed = weights_df.rename(columns={'True DGP (p0)': '$p_0$', 'Sample Size (T)': '$T$'})

        # Pull expected lag for OLS BMA (same BIC weights apply to BMA BVAR-WN)
        bma_exp_lags = (df_renamed[df_renamed['Estimator'] == 'OLS BMA (BIC-W)']
                        [['$p_0$', '$T$', 'Mean Evaluated Lag']].copy())
        bma_exp_lags['Estimator'] = 'Hybrid-BMA (OLS W)'

        dist_df = pd.merge(
            weights_renamed[weights_renamed['Estimator'] == 'Hybrid-BMA (OLS W)'],
            bma_exp_lags, on=['$p_0$', '$T$', 'Estimator'], how='left'
        )
        dist_df_p4 = dist_df[dist_df['$p_0$'] == 4].set_index(['Estimator', '$T$']).drop(columns=['$p_0$'])

        p_cols = [c for c in [f"p={i}" for i in range(1, 13)] if c in dist_df_p4.columns]
        dist_df_p4 = dist_df_p4[['Mean Evaluated Lag'] + p_cols].rename(columns={'Mean Evaluated Lag': '$E[p]$'})

        dist_df_str = dist_df_p4.copy()
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
            caption="Posterior Lag-Weight Distribution for BMA ($p_0 = 4$). OLS BMA and BMA BVAR-WN share identical BIC-based lag weights.",
            label="tab:bma_posterior", multirow_align="t"
        ).replace('Hybrid-BMA (OLS W)', 'BMA (BIC-W)')

        with open(os.path.join(tables_dir, "Table4_BMA_Distribution.tex"), "w") as f:
            f.write(latex4)

    # -------------------------------------------------------------------
    # 4. FIGURES
    # -------------------------------------------------------------------
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # ------------------------------------------------------------------
    # Figure 1: Bias-Variance Tradeoff Curve
    # Continuous curve (30 tau grid) + vertical markers at the 7 discrete
    # taus we actually estimate in the simulation.
    # ------------------------------------------------------------------
    if os.path.exists(tradeoff_path):
        tradeoff_df = pd.read_csv(tradeoff_path)
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.lineplot(data=tradeoff_df, x='Tau', y='Rel_MSE', hue='T', marker='o',
                     palette="Set1", linewidth=2.5, ax=ax)

        # BIC baselines
        for T_val, ls, lbl in [(96, '--', 'SIC Baseline ($T=96$)'), (480, ':', 'SIC Baseline ($T=480$)')]:
            row = df[(df['True DGP (p0)'] == 4) & (df['Sample Size (T)'] == T_val) & (df['Estimator'] == 'SIC (BIC)')]
            if len(row):
                color = '#E41A1C' if T_val == 96 else '#377EB8'
                ax.axhline(row['Geom Mean MSE Ratio'].values[0], color=color, linestyle=ls,
                           alpha=0.7, label=lbl)

        # Vertical markers at the 7 discrete taus we estimate
        for tau in TAU_DISCRETE:
            ax.axvline(tau, color='gray', linestyle=':', linewidth=0.9, alpha=0.5)
        # Dummy line for legend entry
        ax.axvline(TAU_DISCRETE[0], color='gray', linestyle=':', linewidth=0.9, alpha=0.5,
                   label='Estimated $\\tau$ grid')

        ax.set_xlabel(r'Shrinkage Parameter ($\tau$)')
        ax.set_ylabel('Relative Mean Squared Error')
        ax.set_title(r'Bias-Variance Tradeoff Curve ($p_0 = 4$)')
        ax.legend(title='Sample Size')
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "Fig1_Tradeoff_Curve.pdf"))
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: MDD-selected tau distribution across sample sizes
    # (replaces the old "asymptotic relaxation" of optimised tau)
    # ------------------------------------------------------------------
    if os.path.exists(raw_taus_path):
        raw_taus = pd.read_csv(raw_taus_path)
        if 'MDD_Tau' in raw_taus.columns:
            raw_taus_p4 = raw_taus[raw_taus['p0'] == 4]
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=raw_taus_p4, x='T', y='MDD_Tau', hue='T',
                        palette='Blues', legend=False, showfliers=False, ax=ax)
            sns.stripplot(data=raw_taus_p4, x='T', y='MDD_Tau',
                          color='black', alpha=0.3, jitter=True, ax=ax)
            ax.set_yticks(TAU_DISCRETE)
            ax.set_xlabel('Sample Size ($T$)')
            ax.set_ylabel(r'MDD-Selected Shrinkage Parameter ($\tau$)')
            ax.set_title(r'MDD $\tau$ Selection Distribution ($p_0 = 4$)')
            fig.tight_layout()
            fig.savefig(os.path.join(figures_dir, "Fig2_Asymptotic_Relaxation.pdf"))
            plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: MDD Surface
    # ------------------------------------------------------------------
    if os.path.exists(mdd_path):
        mdd_df = pd.read_csv(mdd_path)
        max_idx = mdd_df['MDD'].idxmax()
        max_tau = mdd_df['Tau'][max_idx]
        max_mdd = mdd_df['MDD'][max_idx]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.lineplot(data=mdd_df, x='Tau', y='MDD', color='purple', linewidth=2.5, ax=ax)
        ax.axvline(max_tau, color='black', linestyle='--', alpha=0.6,
                   label=f'Optimal $\\tau \\simeq {max_tau:.2f}$')
        ax.plot(max_tau, max_mdd, 'ko', markersize=8)
        ax.set_xlabel(r'Candidate Shrinkage Parameter ($\tau$)')
        ax.set_ylabel('Marginal Data Density (Log Likelihood)')
        ax.set_title('Objective Function Surface ($T = 480$)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "Fig3_MDD_Surface.pdf"))
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4: Risk Hedging Scatter — BMA BVAR-WN (τ=0.20) vs BIC
    # ------------------------------------------------------------------
    if os.path.exists(iter_mse_path):
        iter_df = pd.read_csv(iter_mse_path)
        scatter_df = iter_df[(iter_df['p0'] == 4) & (iter_df['T'] == 96)].dropna(
            subset=['BMA_Rel_MSE', 'BIC_Rel_MSE'])

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.scatterplot(data=scatter_df, x='BMA_Rel_MSE', y='BIC_Rel_MSE',
                        alpha=0.6, color='#2CA02C', edgecolor='black', s=60, ax=ax)

        lo = min(scatter_df['BMA_Rel_MSE'].min(), scatter_df['BIC_Rel_MSE'].min()) * 0.95
        hi = max(scatter_df['BMA_Rel_MSE'].max(), scatter_df['BIC_Rel_MSE'].max()) * 1.05
        ax.plot([lo, hi], [lo, hi], color='red', linestyle='--', linewidth=2,
                label='Equal Performance ($y=x$)')
        ax.fill_between([lo, hi], [lo, hi], hi, color='green', alpha=0.05,
                        label='BMA Outperforms')

        ax.set_xlim(left=max(0.5, lo))
        ax.set_ylim(bottom=max(0.5, lo))
        ax.set_xlabel(r'BMA BVAR-WN ($\tau=0.20$) Relative MSE')
        ax.set_ylabel('Hard Selection (BIC) Relative MSE')
        ax.set_title(r'Catastrophe Avoidance: BMA vs.~Hard Selection ($p_0=4,\ T=96$)')
        ax.legend(loc='upper left')
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "Fig4_Risk_Hedging_Scatter.pdf"))
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 5: BMA Posterior Weight Heatmap — BMA (BIC-W)
    # ------------------------------------------------------------------
    if weights_df is not None:
        heat_df = weights_df[
            (weights_df['Estimator'] == 'BMA (BIC-W)') &
            (weights_df['True DGP (p0)'] == 4)
        ].copy().set_index('Sample Size (T)')

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

        ax.set_title('Posterior Weight Evolution: BMA (BIC-W) ($p_0=4$)')
        ax.set_xlabel('Sample Size ($T$)')
        ax.set_ylabel('Candidate Lag Order ($p$)')
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "Fig5_BMA_Heatmap.pdf"))
        plt.close(fig)

    print("\n" + "="*80)
    print(" SUCCESS: Tables and Figures generated!")
    print(f" Tables saved to:  {tables_dir}")
    print(f" Figures saved to: {figures_dir}")
    print("="*80)

if __name__ == '__main__':
    main()

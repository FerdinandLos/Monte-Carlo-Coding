import pandas as pd
import os

def main():
    # -------------------------------------------------------------------
    # 1. ROBUST PATH RESOLUTION
    # -------------------------------------------------------------------
    current_script_location = os.path.dirname(os.path.abspath(__file__))
    root_project_folder = os.path.dirname(current_script_location)
    
    KM_FOLDER = 'Kilian and Murphy (2014)'
    km_base = os.path.join(root_project_folder, KM_FOLDER)

    # -------------------------------------------------------------------
    # 2. LOAD THE MASTER DATAFRAME
    # -------------------------------------------------------------------
    filename = "Master_Final_SVAR_Comparison_iters100_draws50.csv"
    read_path = os.path.join(km_base, 'Results', filename)
    
    if not os.path.exists(read_path):
        print(f"\n[ERROR] Could not find the CSV file at:\n{read_path}")
        return
            
    df = pd.read_csv(read_path)

    # -------------------------------------------------------------------
    # 3. RENAME COLUMNS, MAP PLACEHOLDERS & PIVOT
    # -------------------------------------------------------------------
    df = df.rename(columns={
        'True DGP (p0)': '$p_0$',
        'Sample Size (T)': '$T$'
    })

    # We map the estimators to temporary placeholders so Pandas can pivot safely.
    # We will replace these with multi-line LaTeX syntax after the table is built.
    estimator_mapping = {
        'BVAR-Minn (Tight tau=0.05)': 'BVAR_TIGHT_05',
        'BVAR-Minn (Std tau=0.20)': 'BVAR_STD_20',
        'BVAR-Minn (Loose tau=0.50)': 'BVAR_LOOSE_50',
        'BVAR-Conj (Std tau=0.20)': 'BVAR_C_20',
        'BVAR-BIC (Conj.)': 'BVAR-BIC',
        'BVAR-BMA (Conj.)': 'BVAR-BMA'
    }
    df['Estimator'] = df['Estimator'].replace(estimator_mapping)

    pivot_df = df.pivot(
        index=['$p_0$', '$T$'], 
        columns='Estimator', 
        values='Geom Mean MSE Ratio'
    )

    # -------------------------------------------------------------------
    # 4. PAPER 1: THE SHRINKAGE STRATEGY (Priors vs. Data)
    # -------------------------------------------------------------------
    cols_paper1 = [
        'AIC', 'AICc', 'SIC (BIC)', 'HQC', 
        'BVAR_TIGHT_05', 
        'BVAR_STD_20', 
        'BVAR_LOOSE_50', 
        'BVAR_C_20'
    ]
    
    col_format_1 = "ll" + "c" * len(cols_paper1)

    latex_paper1 = (
        pivot_df[cols_paper1].style
        .format("{:.3f}")
        .highlight_min(axis=1, props='textbf:--rwrap;')
        .to_latex(
            column_format=col_format_1,
            position="htbp",
            position_float="centering",
            hrules=True,
            caption="Relative Mean Squared Error: Hard Selection (ICs) vs. Bayesian Shrinkage. Bold values indicate the best performing estimator for a given Sample Size.",
            label="tab:shrinkage_comparison",
            multirow_align="t"
        )
    )
    
    # 1. Rotate table sideways
    latex_paper1 = latex_paper1.replace('\\begin{table}', '\\begin{sidewaystable}').replace('\\end{table}', '\\end{sidewaystable}')
    
    # 2. Inject multi-line LaTeX cells for the column headers (forces 2 rows, drops "std/loose")
    latex_paper1 = latex_paper1.replace(
        'BVAR_TIGHT_05', r'\begin{tabular}{@{}c@{}}BVAR \\ ($\tau=0.05$)\end{tabular}'
    ).replace(
        'BVAR_STD_20', r'\begin{tabular}{@{}c@{}}BVAR \\ ($\tau=0.20$)\end{tabular}'
    ).replace(
        'BVAR_LOOSE_50', r'\begin{tabular}{@{}c@{}}BVAR \\ ($\tau=0.50$)\end{tabular}'
    ).replace(
        'BVAR_C_20', r'\begin{tabular}{@{}c@{}}BVAR-C \\ ($\tau=0.20$)\end{tabular}'
    )

    # -------------------------------------------------------------------
    # 5. PAPER 2: THE UNCERTAINTY STRATEGY (Selection vs. Averaging)
    # -------------------------------------------------------------------
    cols_paper2 = [
        'AIC', 'AICc', 'SIC (BIC)', 'HQC', 
        'OLS BMA (BIC-W)', 
        'BVAR-BIC', 
        'BVAR-BMA'
    ]
    
    col_format_2 = "ll" + "c" * len(cols_paper2)

    latex_paper2 = (
        pivot_df[cols_paper2].style
        .format("{:.3f}")
        .highlight_min(axis=1, props='textbf:--rwrap;')
        .to_latex(
            column_format=col_format_2,
            position="htbp",
            position_float="centering",
            hrules=True,
            caption="Relative Mean Squared Error: Hard Selection (ICs) vs. Bayesian Model Averaging (BMA). Bold values indicate the best performing estimator for a given Sample Size.",
            label="tab:uncertainty_comparison",
            multirow_align="t"
        )
    )
    
    # Rotate table sideways
    latex_paper2 = latex_paper2.replace('\\begin{table}', '\\begin{sidewaystable}').replace('\\end{table}', '\\end{sidewaystable}')

    # -------------------------------------------------------------------
    # 6. EXPORT TO ROOT/Writing/Tables
    # -------------------------------------------------------------------
    save_dir = os.path.join(root_project_folder, "Writing", "Tables")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path_1 = os.path.join(save_dir, "Table_Paper1_Shrinkage.tex")
    save_path_2 = os.path.join(save_dir, "Table_Paper2_Uncertainty.tex")
    
    with open(save_path_1, "w") as f:
        f.write(latex_paper1)
        
    with open(save_path_2, "w") as f:
        f.write(latex_paper2)
        
    print("\n" + "="*80)
    print(" SUCCESS: Tables generated in the root 'Writing/Tables' folder!")
    print(f" Paper 1 (Shrinkage):   {save_path_1}")
    print(f" Paper 2 (Uncertainty): {save_path_2}")
    print("="*80)

if __name__ == '__main__':
    main()
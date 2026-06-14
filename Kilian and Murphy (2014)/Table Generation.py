import pandas as pd
import os

def main():
    # -------------------------------------------------------------------
    # 1. ROBUST PATH RESOLUTION
    # -------------------------------------------------------------------
    # Find the folder where THIS script is located
    current_script_location = os.path.dirname(os.path.abspath(__file__))
    
    # Move UP one level to the root project folder (parent of KM 2014 folder)
    root_project_folder = os.path.dirname(current_script_location)
    
    # Keep KM_FOLDER logic for finding the CSV
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
    # 3. PIVOT THE DATAFRAME FOR ACADEMIC LAYOUT
    # -------------------------------------------------------------------
    pivot_df = df.pivot(
        index=['True DGP (p0)', 'Sample Size (T)'], 
        columns='Estimator', 
        values='Geom Mean MSE Ratio'
    )

    # -------------------------------------------------------------------
    # 4. GENERATE LATEX STRINGS
    # -------------------------------------------------------------------
    cols_paper1 = ['AIC', 'SIC (BIC)', 'HQC', 'BVAR (Minn.)', 'BVAR (Conj.)']
    latex_paper1 = pivot_df[cols_paper1].style.format("{:.3f}").to_latex(
        column_format="llccccc",
        position="htbp",
        position_float="centering",
        hrules=True,
        caption="Relative Mean Squared Error: ICs vs. BVARs.",
        label="tab:bvar_vs_ic",
        multirow_align="t"
    )

    cols_paper2 = ['AIC', 'SIC (BIC)', 'HQC', 'BMA (BIC-Weighted)']
    latex_paper2 = pivot_df[cols_paper2].style.format("{:.3f}").to_latex(
        column_format="llcccc",
        position="htbp",
        position_float="centering",
        hrules=True,
        caption="Relative Mean Squared Error: ICs vs. BMA.",
        label="tab:bma_vs_ic",
        multirow_align="t"
    )

    # -------------------------------------------------------------------
    # 5. EXPORT TO ROOT/Writing/Tables
    # -------------------------------------------------------------------
    # Now save relative to the ROOT project folder
    save_dir = os.path.join(root_project_folder, "Writing", "Tables")
    
    os.makedirs(save_dir, exist_ok=True)
    
    save_path_1 = os.path.join(save_dir, "Table_Paper1_BVARs.tex")
    save_path_2 = os.path.join(save_dir, "Table_Paper2_BMA.tex")
    
    with open(save_path_1, "w") as f:
        f.write(latex_paper1)
        
    with open(save_path_2, "w") as f:
        f.write(latex_paper2)
        
    print("\n" + "="*80)
    print(" SUCCESS: Tables generated in the root 'Writing/Tables' folder!")
    print(f" Paper 1: {save_path_1}")
    print(f" Paper 2: {save_path_2}")
    print("="*80)

if __name__ == '__main__':
    main()
import pandas as pd
import os

# =====================================================================
# SIMULATION CONFIGURATION
# =====================================================================
ITERS = 500  
DRAWS = 1000
FILE_SUFFIX = f"iters{ITERS}_draws{DRAWS}.csv"
COMBINED_SUFFIX = f"iters{ITERS}_draws{DRAWS}_comb.csv"

def combine_results():
    print("\n" + "="*80)
    print(f" COMBINING & SORTING SCRIPT RESULTS FOR: {FILE_SUFFIX}")
    print("="*80)

    # -------------------------------------------------------------------
    # 1. ROBUST PATH RESOLUTION
    # -------------------------------------------------------------------
    current_script_location = os.path.dirname(os.path.abspath(__file__))
    root_project_folder = os.path.dirname(current_script_location)

    KM_FOLDER = 'Kilian and Murphy (2014)'
    km_base = os.path.join(root_project_folder, KM_FOLDER)
    results_dir = os.path.join(km_base, 'Results')

    # -------------------------------------------------------------------
    # 2. DEFINE BASE FILE NAMES
    # -------------------------------------------------------------------
    base_file_names = [
        "Final_SVAR_Comparison",
        "BMA_Weights",
        "Raw_MDD_Tau",
        "Raw_MDD_Tau_p0",
        "Tradeoff_Curve",
        "MDD_Surface",
        "Iteration_MSEs",
        "Expected_Tau_given_p"
    ]

    # -------------------------------------------------------------------
    # 3. LOOP, STACK, SORT, AND SAVE
    # -------------------------------------------------------------------
    for base in base_file_names:
        # Construct exact file names
        file1_name = f"Master_{base}_{FILE_SUFFIX}"
        file2_name = f"Master2_{base}_{FILE_SUFFIX}"
        out_name   = f"Master_{base}_{COMBINED_SUFFIX}"

        # Construct full paths
        file1_path = os.path.join(results_dir, file1_name)
        file2_path = os.path.join(results_dir, file2_name)
        out_path   = os.path.join(results_dir, out_name)

        if os.path.exists(file1_path) and os.path.exists(file2_path):
            print(f"Processing {base}...")
            
            # Read both CSVs
            df1 = pd.read_csv(file1_path)
            df2 = pd.read_csv(file2_path)

            # Combine them vertically
            df_combined = pd.concat([df1, df2], ignore_index=True)

            # ---------------------------------------------------------------
            # NEW: SORTING LOGIC
            # ---------------------------------------------------------------
            if base in ["Final_SVAR_Comparison", "BMA_Weights"]:
                # Sort by 'Sample Size (T)' then 'True DGP (p0)'
                df_combined.sort_values(
                    by=['Sample Size (T)', 'True DGP (p0)'], 
                    ascending=[True, True], 
                    inplace=True
                )
            else:
                # Sort by 'T' then 'p0' for the rest
                # Using a try-except block just in case any file has slightly different names
                try:
                    df_combined.sort_values(
                        by=['T', 'p0'], 
                        ascending=[True, True], 
                        inplace=True
                    )
                except KeyError as e:
                    print(f"  -> [Warning] Could not sort {base} by T and p0. Check column names. Error: {e}")

            # Save the combined and sorted dataframe
            df_combined.to_csv(out_path, index=False)
            print(f"  -> Sorted and saved as: {out_name}")
            
        else:
            print(f"\n[!] SKIPPED {base}: Missing source files.")
            if not os.path.exists(file1_path): print(f"    Missing: {file1_name}")
            if not os.path.exists(file2_path): print(f"    Missing: {file2_name}")

if __name__ == "__main__":
    combine_results()
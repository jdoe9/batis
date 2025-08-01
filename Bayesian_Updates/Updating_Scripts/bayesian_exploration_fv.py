import numpy as np
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Script to run bayesian updates from prior predictions (Fixed Variance Baseline)")
parser.add_argument("--test_set", type=str, required=True, help="Path Test Set (expecting .csv format))")
parser.add_argument("--tau", type=int, default=1, help="Scaling parameter for prior")
parser.add_argument("--mean_folder", type=str, required=True, help="Path to folder containing prior means (expecting .npy format)")
parser.add_argument("--er_folder", type=str, required=True, help="Path to folder containing ground truth encounter rates (expecting .csv format)")
parser.add_argument("--chk_folder", type=str, required=True, help="Path to folder containing hotspots checklists (expecting .csv format)")
parser.add_argument("--n_updates", action=int, default=10, help="Number of times we want to update the posterior distributions")
parser.add_argument("--res_folder", type=str, default="res_bayesian", help="Name of folder in which we want to store the results")

def sample_numbers(x, seed=42):
    """
    x : number of checklists to sample 
    Returns :
        list of int corresponding to index of checklists to use for the bayesian updates in folder
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, x, size=10)

def initialize_beta_parameters(base_preds, tau=1):
    """
    base_preds: array of shape (N,) -> predicted encounter rates from base model (for a given hotspot)
    tau: Scaling parameter (optional)
    Returns:
       alphas, betas: arrays of shape (N,) initializing the prior distribution
    """

    lambda_ = tau * base_preds * (1 - base_preds) 

    alphas = lambda_ * base_preds
    betas  = lambda_ * (1 - base_preds)

    return alphas, betas

def update_beta_binomial(alphas, betas, checklist):
    """
    alphas, betas: arrays of shape (N,) representing parameters of prior distribution (for a given hotspot)
    checklist: array of shape (N,) -> 1 if species i is present, 0 if absent.
    
    Updates alphas, betas in-place, since each species i sees c_i in {0,1}.
    """

    alphas += checklist
    betas  += (1 - checklist)

def posterior_mean(alphas, betas):
    """
    Returns the posterior mean for each species.
    """

    p_mean = alphas / (alphas + betas)
    if np.any(p_mean < 0):
        # The posterior mean can't be negative ! This exception should never happen as edge cases for prior variance are handled
        raise ValueError("Posterior mean contains one or more non-positive elements.")
    return p_mean

# ---------------------------------------------
# Example usage: iterative updates per checklist
# ---------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    results_folder = args.res_folder
    tau = args.tau
    base_mean_folder = args.mean_folder
    encounter_rate_folder = args.er_folder
    checklists_folder = args.chk_folder
    n_checklists = args.n_updates

    df_sorted = pd.read_csv(args.test_set)
    n_hotspots = len(df_sorted) 
    hotspot_id_list = list(df_sorted['hotspot_id'])


    for i in range(len(hotspot_id_list)):
        hotspot_id = hotspot_id_list[i]
        print(hotspot_id)

        base_predictions = np.load(f"{base_mean_folder}/{hotspot_id}.npy")
                
        enc_rates = pd.read_csv(f"{encounter_rate_folder}/{hotspot_id}.csv")
        birds_list = list(enc_rates['ebird_cord'])
        n_checklists = enc_rates.iloc[0]['num_complete_checklists']

        # Initialize Beta priors from base model
        alphas, betas = initialize_beta_parameters(base_predictions, tau=tau)

    
        # Sample n_checklists to perform bayesian updates with 
        all_checklists = os.listdir(f"{checklists_folder}/{hotspot_id}/")
        ids_to_pick = sample_numbers(n_checklists, seed=42)
        df_list = [pd.read_csv(f"{checklists_folder}/{hotspot_id}//{id}.csv") for id in ids_to_pick]

        # Storing observations array
        checklists_array = []
        for j in range(len(df_list)):
            checklists_array.append(df_list[j]['is_observed'])

        # For each checklist, perform the bayesian update
        for j in range(1, len(df_list)+1):
            os.makedirs(f"{results_folder}/step_{j}", exist_ok=True) # Create subfolder for update j
            checklist_npy = np.array(checklists_array[j-1])
            update_beta_binomial(alphas, betas, checklist_npy)
            p_means = posterior_mean(alphas, betas, base_predictions)

            df_dict = {"ebird_cord":birds_list, "encounter_rate":p_means}
            df_dict = pd.DataFrame.from_dict(df_dict)
            df_dict.to_csv(f"{results_folder}/step_" + str(j) + "/" + hotspot_id + ".csv", index=False) # Save posterior predictions iteratively
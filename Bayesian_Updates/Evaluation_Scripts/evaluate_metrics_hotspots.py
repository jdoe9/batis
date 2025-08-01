import torch
import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Script to evaluate performance across hotspots and individual birds")
parser.add_argument("--test_set", type=str, required=True, help="Path Test Set (expecting .csv format))")
parser.add_argument("--mean_folder", type=str, required=True, help="Path to folder containing prior means (expecting .npy format)")
parser.add_argument("--var_folder", type=str, required=True, help="Path to folder containing prior variance (expecting .npy format)")
parser.add_argument("--er_folder", type=str, required=True, help="Path to folder containing ground truth encounter rates (expecting .csv format)")
parser.add_argument("--chk_folder", type=str, required=True, help="Path to folder containing hotspots checklists (expecting .csv format)")
parser.add_argument("--n_updates", action=int, default=10, help="Number of times we want to update the posterior distributions")
parser.add_argument("--res_folder", type=str, default="res_bayesian", help="Name of folder in which we want to store the results")

def custom_topk_single(prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Compute the fraction of indices that match between the top-k of `prediction`
    and the top-k of `ground_truth`, where k is the number of non-zero entries
    in `ground_truth`.

    Args:
        prediction (torch.Tensor): 1D tensor of predicted scores.
        ground_truth (torch.Tensor): 1D tensor of ground-truth scores/labels
                                     (often binary, but not necessarily).

    Returns:
        float: The fraction of overlap between the top-k predicted indices
               and top-k ground-truth indices. Returns 0.0 if there are
               no non-zero entries in the ground_truth.
    """
    # Ensure both have the same shape
    assert prediction.shape == ground_truth.shape, "Shapes must match!"

    # Determine k based on the number of non-zero elements in ground_truth
    k = torch.count_nonzero(ground_truth).item()
    
    # If k == 0, there is nothing to compare; return 0
    if k == 0:
        return 0.0

    # Get the indices of the top-k predictions
    _, pred_indices = torch.topk(prediction, k=k)
    
    # Get the indices of the top-k ground_truth
    _, gt_indices = torch.topk(ground_truth, k=k)
    
    # Compute how many indices are in both sets (the overlap)
    # One straightforward way is to convert indices to sets and compute intersection
    pred_set = set(pred_indices.tolist())
    gt_set   = set(gt_indices.tolist())
    overlap_count = len(pred_set.intersection(gt_set))
    
    # Fraction of top-k matches
    overlap_fraction = overlap_count / k
    return overlap_fraction

def custom_top10_single(target: torch.Tensor, pred: torch.Tensor) -> float:
    """
    For a single (target, pred) pair, compute the fraction of overlapping indices
    in the top-10 predictions and top-(k or 10) of the target, where k is the
    number of non-zero entries in `target`.

    - If k >= 10, we compare top-10 of both `pred` and `target`.
    - If k < 10, we still take top-10 of `pred`, but only top-k from `target`.
      The score is then (#overlapping indices) / 10 if k >= 10, or (#overlapping indices) / k if k < 10.
    - If k == 0, we return 0.0 (no non-zero entries to compare).

    Args:
        target (torch.Tensor): 1D ground-truth tensor.
        pred   (torch.Tensor): 1D prediction tensor (same shape as `target`).

    Returns:
        float: The fraction of matching indices among the chosen top elements.
    """
    # Sanity check: both must be 1D and same shape
    assert target.shape == pred.shape, "Target and pred must have the same shape"
    assert len(target.shape) == 1, "This function expects 1D tensors"

    # Number of non-zero entries in the ground truth
    k = torch.count_nonzero(target).item()
    if k == 0:
        # No non-zero entries => no possible matches
        return 0.0

    # If k >= 10, compare top-10 of both
    # If k < 10, compare top-10 of pred with top-k of target
    if k >= 10:
        _, i_pred = torch.topk(pred,   k=10)
        _, i_targ = torch.topk(target, k=10)
        denom = 10
    else:
        _, i_pred = torch.topk(pred,   k=10)
        _, i_targ = torch.topk(target, k=k)
        denom = k

    # Convert indices to sets and compute overlap
    pred_set = set(i_pred.tolist())
    targ_set = set(i_targ.tolist())
    overlap  = len(pred_set.intersection(targ_set))

    # Fraction of overlap depends on whether k >= 10
    overlap_fraction = overlap / denom
    return overlap_fraction

def custom_top30_single(target: torch.Tensor, pred: torch.Tensor) -> float:
    """
    For a single (target, pred) pair, compute the fraction of overlapping indices
    in the top-10 predictions and top-(k or 10) of the target, where k is the
    number of non-zero entries in `target`.

    - If k >= 10, we compare top-10 of both `pred` and `target`.
    - If k < 10, we still take top-10 of `pred`, but only top-k from `target`.
      The score is then (#overlapping indices) / 10 if k >= 10, or (#overlapping indices) / k if k < 10.
    - If k == 0, we return 0.0 (no non-zero entries to compare).

    Args:
        target (torch.Tensor): 1D ground-truth tensor.
        pred   (torch.Tensor): 1D prediction tensor (same shape as `target`).

    Returns:
        float: The fraction of matching indices among the chosen top elements.
    """
    # Sanity check: both must be 1D and same shape
    assert target.shape == pred.shape, "Target and pred must have the same shape"
    assert len(target.shape) == 1, "This function expects 1D tensors"

    # Number of non-zero entries in the ground truth
    k = torch.count_nonzero(target).item()
    if k == 0:
        # No non-zero entries => no possible matches
        return 0.0

    # If k >= 10, compare top-10 of both
    # If k < 10, compare top-10 of pred with top-k of target
    if k >= 30:
        _, i_pred = torch.topk(pred,   k=30)
        _, i_targ = torch.topk(target, k=30)
        denom = 30
    else:
        _, i_pred = torch.topk(pred,   k=30)
        _, i_targ = torch.topk(target, k=k)
        denom = k

    # Convert indices to sets and compute overlap
    pred_set = set(i_pred.tolist())
    targ_set = set(i_targ.tolist())
    overlap  = len(pred_set.intersection(targ_set))

    # Fraction of overlap depends on whether k >= 10
    overlap_fraction = overlap / denom
    return overlap_fraction

def mae_single(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes the Mean Absolute Error (MAE) for a single (prediction, target) pair.

    Args:
        pred (torch.Tensor): The predicted values.
        target (torch.Tensor): The true/ground-truth values.

    Returns:
        float: The MAE for these two tensors.
    """
    # Ensure the shapes match
    assert pred.shape == target.shape, "Prediction and target must have the same shape"

    # Compute the absolute error and then the mean
    error = (pred - target).abs().mean()

    # Convert from a tensor to a Python float
    return error.item()

def mse_single(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes the Mean Squared Error (MSE) for a single (prediction, target) pair.

    Args:
        pred (torch.Tensor): The predicted values.
        target (torch.Tensor): The ground-truth values.

    Returns:
        float: The MSE for these two tensors.
    """
    # Check that they have the same shape
    assert pred.shape == target.shape, "Prediction and target must have the same shape"

    # Compute (pred - target)^2 and take the mean
    error = (pred - target).pow(2).mean()

    # Convert from a tensor to a Python float
    return error.item()


args = parser.parse_args()

test_data = pd.read_csv("test_filtered.csv")
hotspots_list = list(test_data['hotspot_id'])
seed_id = "mvn_877"
variance_id = "mvn_var"
steps_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for step_stage in steps_range:
    predictions_folder = f"/Users/Desktop/Testing_Env/evaluate_results/updates_res/{seed_id}/{variance_id}/step_{step_stage}/"
    target_folder = "/Users/Desktop/Bayesian_SDM_Project/Encounter_Rates/Concat_April5/"

    results_by_hotspot_folder = f"results_by_hotspot/{seed_id}/{variance_id}/step_{step_stage}"
    os.makedirs(results_by_hotspot_folder, exist_ok=True)
    results_by_hotspot_folder += "/"

    results_by_bird_folder = f"results_by_bird/{seed_id}/{variance_id}/step_{step_stage}"
    os.makedirs(results_by_bird_folder, exist_ok=True)
    results_by_bird_folder += "/"

    hotspot_df = []
    mse_df = []
    mae_df = []
    topk_df = []
    top10_df = []
    top30_df = []

    list_birds = 0
    for hotspot in hotspots_list:
        print(hotspot)
        preds = pd.read_csv(predictions_folder + hotspot + ".csv")
        preds = np.array(list(preds['encounter_rate']))
    
        target_df = pd.read_csv(target_folder + hotspot + ".csv")
        target_array = np.array(list(target_df['is_observed']))
        list_birds = list(target_df['ebird_cord'])

        prediction = torch.tensor(preds)
        ground_truth = torch.tensor(target_array)

        hotspot_df.append(hotspot)
        mse_df.append(mse_single(prediction, ground_truth))
        mae_df.append(mae_single(prediction, ground_truth))
        top10_df.append(custom_top10_single(ground_truth, prediction))
        top30_df.append(custom_top30_single(ground_truth, prediction))
        topk_df.append(custom_topk_single(prediction, ground_truth))

    dict_hotspot = {"hotspot_id":hotspot_df, "mse":mse_df, "mae":mae_df, "topk":topk_df, "top10":top10_df, "top30":top30_df}
    df_hotspot = pd.DataFrame.from_dict(dict_hotspot)
    df_hotspot.to_csv(results_by_hotspot_folder + "hotspot_results.csv", index=False)

    for i in range(len(list_birds)):
        bird_id = list_birds[i]
        print(bird_id)

        hotspot_bird = []
        mae_bird = []
        mse_bird = []
        for hotspot in hotspots_list:
            hotspot_bird.append(hotspot)
            preds = pd.read_csv(predictions_folder + hotspot + ".csv")
            preds = list(preds['encounter_rate'])[i]
            preds = np.array([preds])
            
            target_df = pd.read_csv(target_folder + hotspot + ".csv")
            target_array = np.array([list(target_df['is_observed'])[i]])

            prediction = torch.tensor(preds)
            ground_truth = torch.tensor(target_array)
            
            mse_bird.append(mse_single(prediction, ground_truth))
            mae_bird.append(mae_single(prediction, ground_truth))
        
        dict_bird = {"hotspot_id":hotspot_bird, "mse":mse_bird, "mae":mae_bird}
        df_bird = pd.DataFrame.from_dict(dict_bird)
        df_bird.to_csv(results_by_bird_folder + bird_id + ".csv", index=False)


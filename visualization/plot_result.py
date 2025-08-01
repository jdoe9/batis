import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from matplotlib.colors import to_hex

metric_target = "topk"
metric_name = "Top-K"

def get_mean_and_stdev_mae(seeds, mean_0, stdev_0):
    val_arr = []
    for seed in seeds:
        mean_mae = []
        for i in range(1, 11):
            results_by_hotspot_path = f"results_by_hotspot/{seed}/step_{i}/hotspot_results.csv"
            df_res = pd.read_csv(results_by_hotspot_path)
            mean_mae.append(np.sum(df_res['mae'])/len(df_res)*100)
        val_arr.append(mean_mae)
      
    val_arr = np.array(val_arr)
    mean = np.mean(val_arr, axis=0)
    stdev = np.std(val_arr, axis=0)
    mean = np.concatenate([np.array([mean_0]), mean])
    stdev = np.concatenate([np.array([stdev_0]), stdev])
    return mean, stdev

def get_mean_and_stdev_mse(seeds, mean_0, stdev_0):
    val_arr = []
    for seed in seeds:
        mean_mae = []
        for i in range(1, 11):
            results_by_hotspot_path = f"results_by_hotspot/{seed}/step_{i}/hotspot_results.csv"
            df_res = pd.read_csv(results_by_hotspot_path)
            mean_mae.append(np.sum(df_res['mse'])/len(df_res)*100)
        val_arr.append(mean_mae)
      
    val_arr = np.array(val_arr)
    mean = np.mean(val_arr, axis=0)
    stdev = np.std(val_arr, axis=0)
    mean = np.concatenate([np.array([mean_0]), mean])
    stdev = np.concatenate([np.array([stdev_0]), stdev])
    return mean, stdev

def get_mean_and_stdev_top10(seeds, mean_0, stdev_0):
    val_arr = []
    for seed in seeds:
        mean_mae = []
        for i in range(1, 11):
            results_by_hotspot_path = f"results_by_hotspot/{seed}/step_{i}/hotspot_results.csv"
            df_res = pd.read_csv(results_by_hotspot_path)
            mean_mae.append(np.sum(df_res['top10'])/len(df_res)*100)
        val_arr.append(mean_mae)
      
    val_arr = np.array(val_arr)
    mean = np.mean(val_arr, axis=0)
    stdev = np.std(val_arr, axis=0)
    mean = np.concatenate([np.array([mean_0]), mean])
    stdev = np.concatenate([np.array([stdev_0]), stdev])
    return mean, stdev

######################

# FIXED VARIANCE

######################

seeds = ["Kenya/FV1", "Kenya/FV2", "Kenya/FV3"]
mean_fv, stdev_fv = get_mean_and_stdev_mae(seeds, 1.87, 0.04)
mean_mse_fv, stdev_mse_fv = get_mean_and_stdev_mse(seeds, 0.35, 0.01)
mean_top10_fv, stdev_top10_fv = get_mean_and_stdev_top10(seeds, 41.24, 1.82)

######################

# DEEP ENSEMBLES

######################

ensembles_folder = ["Kenya/DeepEns/1", "Kenya/DeepEns/2", "Kenya/DeepEns/3"]

mean_de, stdev_de = get_mean_and_stdev_mae(seeds, 1.81, 0.04)
mean_mse_de, stdev_mse_de = get_mean_and_stdev_mse(seeds, 0.32, 0.01)
mean_top10_de, stdev_top10_de = get_mean_and_stdev_top10(seeds, 44.34, 0.31)

######################

# HISTORICAL VARIANCE

######################

seeds = ["Kenya/HistVar1/1", "Kenya/HistVar1/2", "Kenya/HistVar1/3"]

mean_hist, stdev_hist = get_mean_and_stdev_mae(seeds, 1.87, 0.04)
mean_mse_hist, stdev_mse_hist = get_mean_and_stdev_mse(seeds, 0.35, 0.01)
mean_top10_hist, stdev_top10_hist = get_mean_and_stdev_top10(seeds, 41.24, 1.82)

######################

# MVN NETWORK 

######################

seeds = ["seed_877/mvn_var", "seed_178/mvn_var", "seed_1234/mvn_var"]

mean_mvn, stdev_mvn = get_mean_and_stdev_mae(seeds, 2.04, 0.11)
mean_mse_mvn, stdev_mse_mvn = get_mean_and_stdev_mse(seeds, 0.37, 0.01)
mean_top10_mvn, stdev_top10_mvn = get_mean_and_stdev_top10(seeds, 36.33, 1.07)

######################

# HETREG NETWORK 

######################

seeds = ["Kenya/HetReg/1", "Kenya/HetReg/2", "Kenya/HetReg/3"]

mean_hetreg, stdev_hetreg = get_mean_and_stdev_mae(seeds, 2.04, 0.02)
mean_mse_hetreg, stdev_mse_hetreg = get_mean_and_stdev_mse(seeds, 0.37, 0.01)
mean_top10_hetreg, stdev_top10_hetreg = get_mean_and_stdev_top10(seeds, 36.24, 1.07)

######################

# SHALLOW ENSEMBLES

######################

seeds = ["Kenya/Shallow/1", "Kenya/Shallow/2", "Kenya/Shallow/3"]

mean_shallow, stdev_shallow = get_mean_and_stdev_mae(seeds, 1.73, 0.01)
mean_mse_shallow, stdev_mse_shallow = get_mean_and_stdev_mse(seeds, 0.33, 0.001)
mean_top10_shallow, stdev_top10_shallow = get_mean_and_stdev_top10(seeds, 45.53, 0.26)

######################

# MCD NETWORK 

######################

seeds = ["Kenya/Dropout/1", "Kenya/Dropout/2", "Kenya/Dropout/3"]

mean_mcd, stdev_mcd = get_mean_and_stdev_mae(seeds, 1.91, 0.01)
mean_mse_mcd, stdev_mse_mcd = get_mean_and_stdev_mse(seeds, 0.37, 0.01)
mean_top10_mcd, stdev_top10_mcd = get_mean_and_stdev_top10(seeds, 38.65, 1.19)

##########################

# PLOTTING

##########################

steps = list(range(0, 11))
palette_graphs = sns.color_palette("dark", 7)
hex_colors_graphs = [to_hex(color) for color in palette_graphs]

palette_fill = sns.color_palette("muted", 7)
hex_colors_fill = [to_hex(color) for color in palette_fill]

# Create the figure
fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(2, 3, height_ratios=[0.1, 1])  # Top row for legend (10% height)

# Create subplots in the second row
axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

# Plot on each subplot

ax_count = 0
for ax in axes:
    if ax_count == 0:
        ax.plot(steps, mean_de, "-o", color=palette_graphs[0], linewidth=1.0, label="Deep Ensembles")
        ax.fill_between(steps, (mean_de-stdev_de), (mean_de+stdev_de), color=palette_fill[0], alpha=0.3, lw=0)

        ax.plot(steps, mean_fv, "-o", color=palette_graphs[1], linewidth=1.0, label=r"Fixed Variance ($\sigma^{2} = \mu (1 - \mu) $)")
        ax.fill_between(steps, (mean_fv-stdev_fv), (mean_fv+stdev_fv), color=palette_fill[1], alpha=0.3, lw=0)

        ax.plot(steps, mean_hist, "-o", color=palette_graphs[2], linewidth=1.0, label=r"Historical Variance")
        ax.fill_between(steps, (mean_hist-stdev_hist), (mean_hist+stdev_hist), color=palette_fill[2], alpha=0.3, lw=0)

        ax.plot(steps, mean_mvn, "-o", color=palette_graphs[3], linewidth=1.0, label=r"Mean-Variance Network")
        ax.fill_between(steps, (mean_mvn-stdev_mvn), (mean_mvn+stdev_mvn), color=palette_fill[3], alpha=0.3, lw=0)

        ax.plot(steps, mean_hetreg, "-o", color=palette_graphs[4], linewidth=1.0, label=r"HetReg Network")
        ax.fill_between(steps, (mean_hetreg-stdev_hetreg), (mean_hetreg+stdev_hetreg), color=palette_fill[4], alpha=0.3, lw=0)

        ax.plot(steps, mean_mcd, "-o", color=palette_graphs[5], linewidth=1.0, label=r"Monte-Carlo Dropout")
        ax.fill_between(steps, (mean_mcd-stdev_mcd), (mean_mcd+stdev_mcd), color=palette_fill[5], alpha=0.3, lw=0)

        ax.plot(steps, mean_shallow, "-o", color=palette_graphs[6], linewidth=1.0, label=r"Shallow Ensembles")
        ax.fill_between(steps, (mean_shallow-stdev_shallow), (mean_shallow+stdev_shallow), color=palette_fill[6], alpha=0.3, lw=0)

        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
        ax.set_ylabel("MAE")
        ax.set_xlabel("Number of checklists")

    elif ax_count == 1:
        ax.plot(steps, mean_mse_de, "-o", color=palette_graphs[0], linewidth=1.0, label="Deep Ensembles")
        ax.fill_between(steps, (mean_mse_de-stdev_mse_de), (mean_mse_de+stdev_mse_de), color=palette_fill[0], alpha=0.3, lw=0)

        ax.plot(steps, mean_mse_fv, "-o", color=palette_graphs[1], linewidth=1.0, label=r"Fixed Variance ($\sigma^{2} = 0.25$)")
        ax.fill_between(steps, (mean_mse_fv-stdev_mse_fv), (mean_mse_fv+stdev_mse_fv), color=palette_fill[1], alpha=0.3, lw=0)

        ax.plot(steps, mean_mse_hist, "-o", color=palette_graphs[2], linewidth=1.0, label=r"Historical Variance")
        ax.fill_between(steps, (mean_mse_hist-stdev_mse_hist), (mean_mse_hist+stdev_mse_hist), color=palette_fill[2], alpha=0.3, lw=0)

        ax.plot(steps, mean_mse_mvn, "-o", color=palette_graphs[3], linewidth=1.0, label=r"Mean-Variance Network")
        ax.fill_between(steps, (mean_mse_mvn-stdev_mse_mvn), (mean_mse_mvn+stdev_mse_mvn), color=palette_fill[3], alpha=0.3, lw=0)

        ax.plot(steps, mean_mse_hetreg, "-o", color=palette_graphs[4], linewidth=1.0, label=r"HetReg Network")
        ax.fill_between(steps, (mean_mse_hetreg-stdev_mse_hetreg), (mean_mse_hetreg+stdev_mse_hetreg), color=palette_fill[4], alpha=0.3, lw=0)

        ax.plot(steps, mean_mse_mcd, "-o", color=palette_graphs[5], linewidth=1.0, label=r"Monte-Carlo Dropout")
        ax.fill_between(steps, (mean_mse_mcd-stdev_mse_mcd), (mean_mse_mcd+stdev_mse_mcd), color=palette_fill[5], alpha=0.3, lw=0)

        ax.plot(steps, mean_mse_shallow, "-o", color=palette_graphs[6], linewidth=1.0, label=r"Shallow Ensembles")
        ax.fill_between(steps, (mean_mse_shallow-stdev_mse_shallow), (mean_mse_shallow+stdev_mse_shallow), color=palette_fill[6], alpha=0.3, lw=0)

        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
        ax.set_ylabel("MSE")
        ax.set_xlabel("Number of checklists")

    elif ax_count == 2:
        ax.plot(steps, mean_top10_de, "-o", color=palette_graphs[0], linewidth=1.0, label="Deep Ensembles")
        ax.fill_between(steps, (mean_top10_de-stdev_top10_de), (mean_top10_de+stdev_top10_de), color=palette_fill[0], alpha=0.3, lw=0)

        ax.plot(steps, mean_top10_fv, "-o", color=palette_graphs[1], linewidth=1.0, label=r"Fixed Variance ($\sigma^{2} = 0.25$)")
        ax.fill_between(steps, (mean_top10_fv-stdev_top10_fv), (mean_top10_fv+stdev_top10_fv), color=palette_fill[1], alpha=0.3, lw=0)

        ax.plot(steps, mean_top10_hist, "-o", color=palette_graphs[2], linewidth=1.0, label=r"Historical Variance")
        ax.fill_between(steps, (mean_top10_hist-stdev_top10_hist), (mean_top10_hist+stdev_top10_hist), color=palette_fill[2], alpha=0.3, lw=0)

        ax.plot(steps, mean_top10_mvn, "-o", color=palette_graphs[3], linewidth=1.0, label=r"Mean-Variance Network")
        ax.fill_between(steps, (mean_top10_mvn-stdev_top10_mvn), (mean_top10_mvn+stdev_top10_mvn), color=palette_fill[3], alpha=0.3, lw=0)

        ax.plot(steps, mean_top10_hetreg, "-o", color=palette_graphs[4], linewidth=1.0, label=r"HetReg Network")
        ax.fill_between(steps, (mean_top10_hetreg-stdev_top10_hetreg), (mean_top10_hetreg+stdev_top10_hetreg), color=palette_fill[4], alpha=0.3, lw=0)

        ax.plot(steps, mean_top10_mcd, "-o", color=palette_graphs[5], linewidth=1.0, label=r"Monte-Carlo Dropout")
        ax.fill_between(steps, (mean_top10_mcd-stdev_top10_mcd), (mean_top10_mcd+stdev_top10_mcd), color=palette_fill[5], alpha=0.3, lw=0)

        ax.plot(steps, mean_top10_shallow, "-o", color=palette_graphs[6], linewidth=1.0, label=r"Shallow Ensembles")
        ax.fill_between(steps, (mean_top10_shallow-stdev_top10_shallow), (mean_top10_shallow+stdev_top10_shallow), color=palette_fill[6], alpha=0.3, lw=0)

        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
        ax.set_ylabel("Top-10")
        ax.set_xlabel("Number of checklists")
    ax_count += 1
         
handles, labels = axes[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', title="Models", ncol=4, bbox_to_anchor=(0.5, 0.98), fontsize=12)
legend.get_title().set_fontweight('bold')

# Adjust layout so the legend doesn't overlap
plt.tight_layout(rect=[0, 0, 1, 0.92])

plt.show()

fig.savefig("fig_kenya_appendix_graph.png", dpi=300)
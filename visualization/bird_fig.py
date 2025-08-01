import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import matplotlib.image as mpimg

def create_grid(gdf, cell_size_km=10):
    # Compute bounding box of the GeoDataFrame
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # Convert km to meters
    cell_size_m = cell_size_km * 1000
    
    # Create list of polygon cells
    grid_cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            # Create a polygon covering the cell_size_m x cell_size_m area
            cell_polygon = box(x, y, x + cell_size_m, y + cell_size_m)
            grid_cells.append(cell_polygon)
            y += cell_size_m
        x += cell_size_m

    # Build a GeoDataFrame from the grid cells
    grid_gdf = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=gdf.crs)
    return grid_gdf

kenya = gpd.read_file("Kenya/kenya.shp")
gdf_kenya_utm = kenya.to_crs(epsg=21037)
grid_gdf = create_grid(gdf_kenya_utm, cell_size_km=70)


test_points2 = pd.read_csv("/Users/Desktop/Bayesian_SDM_Project/values_per_region/spepig1.csv")
geometry = []
for i in range(len(test_points2)):
    row = test_points2.iloc[i]
    geometry.append(Point(row['lon'], row['lat']))

vmin = 0
vmax = 1
cmap = cm.get_cmap('YlOrRd')
norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=1)
exp_type = "seed_877/var_025"
seeds = [1, 3, 6, 10]

dataset = []
fig, axes = plt.subplots(3, 5, figsize=(10, 8), constrained_layout=True)


for seed in seeds:
    test_points = pd.read_csv(f"/Users/Desktop/Testing_Env/evaluate_results/results_by_bird/{exp_type}/step_{seed}/spepig1.csv")

    points_gdf = gpd.GeoDataFrame(test_points, geometry=geometry, crs="EPSG:4326")
    gdf_points_utm = points_gdf.to_crs(epsg=21037)

    grid_gdf = create_grid(gdf_kenya_utm, cell_size_km=70)
    grid_clipped = gpd.clip(grid_gdf, gdf_kenya_utm)

    joined = gpd.sjoin(gdf_points_utm, grid_clipped, how="left", predicate="within")
    grid_stats = joined.groupby("index_right")["mae"].mean()
    grid_clipped["mean_value"] = grid_stats
    print(np.min(grid_clipped['mean_value']))
    dataset.append(grid_clipped)    


count = 0
ax = axes[0][0]
img = mpimg.imread('speckled_pigeon.jpg') 
ax.imshow(img)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
for ax, grid_fig in zip(axes[0][1:], dataset):
    gdf_kenya_utm.plot(ax=ax, color="lightgrey", edgecolor="black")
    im = grid_fig.plot(column="mean_value",  ax=ax,  cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    ax.set_title(f"n = {seeds[count]}", fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    count += 1


test_points2 = pd.read_csv("/Users/Bayesian_SDM_Project/values_per_region/graher1.csv")
geometry = []
for i in range(len(test_points2)):
    row = test_points2.iloc[i]
    geometry.append(Point(row['lon'], row['lat']))

dataset = []
for seed in seeds:
    test_points = pd.read_csv(f"Testing_Env/evaluate_results/results_by_bird/{exp_type}/step_{seed}/graher1.csv")

    points_gdf = gpd.GeoDataFrame(test_points, geometry=geometry, crs="EPSG:4326")
    gdf_points_utm = points_gdf.to_crs(epsg=21037)

    grid_gdf = create_grid(gdf_kenya_utm, cell_size_km=70)
    grid_clipped = gpd.clip(grid_gdf, gdf_kenya_utm)

    joined = gpd.sjoin(gdf_points_utm, grid_clipped, how="left", predicate="within")
    grid_stats = joined.groupby("index_right")["mae"].mean()
    grid_clipped["mean_value"] = grid_stats
    print(np.min(grid_clipped['mean_value']))
    dataset.append(grid_clipped)    

count = 0
ax = axes[1][0]
img = mpimg.imread('grey_heron.jpg') 
ax.imshow(img)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
for ax, grid_fig in zip(axes[1][1:], dataset):
    gdf_kenya_utm.plot(ax=ax, color="lightgrey", edgecolor="black")
    im = grid_fig.plot(column="mean_value",  ax=ax,  cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    count += 1


test_points2 = pd.read_csv("/Users/Bayesian_SDM_Project/values_per_region/bagwea1.csv")
geometry = []
for i in range(len(test_points2)):
    row = test_points2.iloc[i]
    geometry.append(Point(row['lon'], row['lat']))


dataset = []
for seed in seeds:
    test_points = pd.read_csv(f"/Users/Testing_Env/evaluate_results/results_by_bird/{exp_type}/step_{seed}/bagwea1.csv")

    points_gdf = gpd.GeoDataFrame(test_points, geometry=geometry, crs="EPSG:4326")
    gdf_points_utm = points_gdf.to_crs(epsg=21037)

    grid_gdf = create_grid(gdf_kenya_utm, cell_size_km=70)
    grid_clipped = gpd.clip(grid_gdf, gdf_kenya_utm)

    joined = gpd.sjoin(gdf_points_utm, grid_clipped, how="left", predicate="within")
    grid_stats = joined.groupby("index_right")["mae"].mean()
    grid_clipped["mean_value"] = grid_stats
    print(np.min(grid_clipped['mean_value']))
    dataset.append(grid_clipped)    

count = 0
ax = axes[2][0]
img = mpimg.imread('baglafecht_weaver.png') 
ax.imshow(img)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
for ax, grid_fig in zip(axes[2][1:], dataset):
    gdf_kenya_utm.plot(ax=ax, color="lightgrey", edgecolor="black")
    im = grid_fig.plot(column="mean_value",  ax=ax,  cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    count += 1

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  

# Add the colorbar at the top
cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks_position('top')
cbar.set_label("Mean Absolute Error")
plt.show()
plt.tight_layout()
fig.savefig("combined_picture.eps", format="eps")
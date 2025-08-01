from sklearn.cluster import DBSCAN
from collections import defaultdict
from copy import deepcopy
import random
import pandas as pd
import os
import json
import random
import numpy as np
import glob
from math import radians
from sklearn.model_selection import train_test_split


RADIUS_EARTH = 6356.7523  # in km, polar radius of Earth

def get_lat_for_distance(d):
    # https://github.com/sustainlab-group/africa_poverty
    '''
    Helper function

    Calculates the degrees latitude for some North-South distance.

    Makes (incorrect) assumption that Earth is a perfect sphere.
    Uses the smaller polar radius (instead of equatorial radius), so
        actual degrees latitude <= returned value

    Args
    - d: numeric, distance in km

    Returns
    - lat: float, approximate degrees latitude
    '''
    lat = d / RADIUS_EARTH  # in radians
    lat = lat * 180.0 / np.pi  # convert to degrees
    return lat


def get_lon_for_distance(lat, d):
    # https://github.com/sustainlab-group/africa_poverty
    '''
    Helper Function
    Calculates the degrees longitude for some East-West distance at a given latitude.

    Makes (incorrect) assumption that Earth is a perfect sphere.
    Uses the smaller polar radius (instead of equatorial radius), so
        actual degrees longitude <= returned value

    Args
    - lat: numeric, latitude in degrees
    - d: numeric, distance in km

    Returns
    - lon: float, approximate degrees longitude
    '''
    lat = np.abs(lat) * np.pi / 180.0  # convert to radians
    r = RADIUS_EARTH * np.cos(lat)  # radius at the given lat
    lon = d / r
    lon = lon * 180.0 / np.pi  # convert to degrees
    return lon


from sklearn.cluster import DBSCAN

def cluster_based_on_dist(X, dist):
    '''
    label the cluster each input belongs to , where the cluster is a group of points in that are not far away from other by  dist km
    each group of points contains at least 2 samples , if you used more than 2 samples=> less number of clusters
    the distance in km is converted to lat, lon using some formulas , then that total distance is approximated as a straight line between lat & lon
    more on #https://jonisalonen.com/2014/computing-distance-between-coordinates-can-be-simple-and-fast/


    Args
    X: array of shape(data_size ,2) contains [lat, lon]
    dist: numeric, maximum distance  in km between points in a cluster

    Returns
    - cluster_labels :array of shape(data_size,) labeling each [lat,lon]
    - clusters_dict  :dict(cluster_label:[indices]) map each label to its list of indices in X
    '''
    clustering = DBSCAN( eps=dist, min_samples=2, metric='haversine').fit(X)
    cluster_labels = clustering.labels_
    # map each label to its indices, each outlier (label of -1) is treated as its own cluster
    loc_indices = defaultdict(list)
    for i, loc in enumerate(X):
        loc_indices[tuple(loc)].append(i)
    clusters_dict = defaultdict(list)
    neg = -1
    for loc, label in zip(X, cluster_labels):
        indices = loc_indices[tuple(loc)]
        if label < 0:
            label = neg
            neg -= 1
        clusters_dict[label].extend(indices)
    return clusters_dict, cluster_labels


def make_splits(splits_names, sizes, clusters_dict):
    '''
    maps each split to its indices with the size of the split , making sure that each cluster is all in one split and not scattered between different splits
    Args
    - splits: list of string, naming splits
    - sizes: list of floats , size of each split
    - clusters_dict : dict(cluster_label:[indices]) map each label to its indices

    Returns
    - splits: dict(split:[indices]) maps split to list of indices
    '''
    splits = defaultdict(list)

    cop_cluster = deepcopy(clusters_dict)
    for s, size in zip(splits_names, sizes):
        while cop_cluster:
            c = random.choice(list(cop_cluster))
            # print(c)
            splits[s].extend(clusters_dict[c])
            cop_cluster.pop(c)

            if len(splits[s]) >= size:
                # print(c)
                break
    return splits


def write_array_text(arr, file):
    with open(file, "w") as txt_file:
        for elem in arr:
            txt_file.write(elem + "\n")

def collapse_group(group):
        first = group.iloc[0].copy()
        first["num_complete_checklists"] = group["num_complete_checklists"].sum()
        return first

if __name__ == "__main__":
    """
    
    MERGING CHECKLISTS THAT ARE WITHIN 111M OF EACH OTHER 
    
    """

    df = pd.read_csv("combined_data.csv")
    # Round coordinates
    df["lat_round"] = df["lat"].round(2)
    df["lon_round"] = df["lon"].round(2)

    # Group by rounded coordinates
    grouped = df.groupby(["lat_round", "lon_round"])
    print(len(grouped))

    result_df = grouped.apply(collapse_group).reset_index(drop=True)
    result_df = result_df.sort_values(by="num_complete_checklists", ascending=False).reset_index(drop=True)
    result_df.to_csv("combined_data.csv")

    """
    
    MAKE SURE WE DON'T INCLUDE HOTSPOTS WITH LESS THAN 15 CHECKLIST IN OUR TEST
    
    """


    df_few_checklists    = result_df[result_df["num_complete_checklists"] < 15]
    df_enough_checklists = result_df[result_df["num_complete_checklists"] >= 15]

    # Save to CSV
    df_few_checklists.to_csv("hotspot_not_to_include_in_test.csv", index=False)
    df_enough_checklists.to_csv("hotspot_to_include_in_test.csv", index=False)


    """
    
    APPLY DBSCAN TO REDUCE AUTOCORRELATION
    
    """

    np.random.seed(0)
    random.seed(0)
    # Reading Hotspot data

    locs = pd.read_csv("hotspot_to_include_in_test.csv")
    locs = locs.sample(frac=1).reset_index(drop=True)

    print(len(locs))
    u = locs[['lat', 'lon']]
    u['lat']= np.radians(u["lat"])
    u['lon']= np.radians(u["lon"])
    # splitting ---
    X = u.values
   
    clusters_dict, cluster_labels = cluster_based_on_dist(u.values, dist=5/RADIUS_EARTH )
    train = 0.4
    valid = 0.1
    splits_names = ['valid','test','train']

    train_size = int(train * len(X))
    val_size = int(valid * len(X))
    test_size = len(locs) - (train_size + val_size)

    print((train_size , val_size, test_size))
    sizes = [val_size, test_size,  train_size]

    splits = make_splits(splits_names, sizes, clusters_dict)
    print([(name, len(splits[name])) for name in splits_names])
    # Write to text files

    # Write to csv files:
    df = locs
    df["split"] = ""
    for name in splits_names:
        df.loc[splits[name], "split"] = name #.reset_index(drop=True)
    df = df.sort_values(by="num_complete_checklists", ascending=False)
    df[df['split'] == 'train'].to_csv('clustered_train_summer.csv', index=False)
    df[df['split'] == 'test'].to_csv('clustered_test_summer.csv', index=False)
    df[df['split'] == 'valid'].to_csv('clustered_val_summer.csv', index=False)

    """
    
    FINAL SPLITS
    
    """

    original_df = pd.read_csv("combined_data.csv")
    df_not_to_include = pd.read_csv("hotspot_not_to_include_in_test.csv")
    df_to_include = pd.read_csv("hotspot_to_include_in_test.csv")

    X_a = pd.read_csv("clustered_train_summer.csv")
    X_b = pd.read_csv("clustered_test_summer.csv")
    X_c = pd.read_csv("clustered_val_summer.csv")

    shuffled_df = df_not_to_include.sample(frac=1).reset_index(drop=True)
    print(len(shuffled_df))

    Y_a, Y_b = train_test_split(shuffled_df, train_size=0.8, test_size=0.2, random_state=42)
    print(len(Y_a))
    print(len(Y_b))

    names_test = list(X_b['hotspot_id']) 
    names_train = list(Y_a['hotspot_id']) + list(X_b['hotspot_id'])
    names_valid = list(Y_b['hotspot_id']) + list(X_c['hotspot_id'])

    print("")
    filtered_test = original_df[original_df['hotspot_id'].isin(names_test)].copy()
    filtered_test['split'] = "test"
    print(len(filtered_test))
    filtered_test.to_csv("test_filtered.csv", index=False)


    filtered_train = original_df[original_df['hotspot_id'].isin(names_train)].copy()
    filtered_train['split'] = "train"
    print(len(filtered_train))
    filtered_train.to_csv("train_filtered.csv", index=False)

    filtered_val = original_df[original_df['hotspot_id'].isin(names_valid)].copy()
    filtered_val['split'] = "valid"
    print(len(filtered_val))
    filtered_val.to_csv("valid_filtered.csv", index=False)
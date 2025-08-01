from __future__ import annotations

import argparse
import functools
import itertools
import json
import multiprocessing as mp
import os
import os.path
from pathlib import Path
import shutil
import csv
import numpy as np
from scipy.interpolate import interp2d
import glob
from tqdm import tqdm

N_CPUS = 3

H, W = 6,6

# Example data with NaN values
x = np.linspace(0, H-1, H)
y = np.linspace(0, W-1, W)
xi, yi = np.meshgrid(x, y)

def bilinear_interpolation(elem, nans):
    # Perform bilinear nterpolation
    f = interp2d(xi[~nans], yi[~nans], elem[~nans], kind='linear', bounds_error=False )
    return f(x, y)


def process_raster(file):
    r = np.load(file)
    u = r.copy()
    for i,elem in enumerate(r): 
        nans = np.isnan(elem)
        nonnan = ~np.isnan(elem)
        if nans.any() and nonnan.any():
                # Define the coordinates of NaN values
                # Create a function for bicubic interpolation with extrapolation
            try:
                arr = bilinear_interpolation(elem, nans)
                arr[~nans] = elem[~nans]
                u[i] = arr
            except:
                #for example when only 2 points are available, interpolation is not possible with linear
                u[i]=r[i]
                print(os.path.basename(file))
    return(u, file)


        
if __name__=="__main__":
    rasters = glob.glob("/Volumes/My Passport/South_Africa/environmental/*")
    print(rasters)
    nan_counter = 0
    for raster_file in rasters:
        basename = os.path.basename(raster_file)
        a = np.load(raster_file)
        has_nan = np.isnan(a).any()
        if has_nan:
            data, _ = process_raster(raster_file)
            np.save(f"/Volumes/My Passport/South_Africa/environmental_filled2/{basename}", data)
            nan_counter += 1
        else:
            np.save(f"/Volumes/My Passport/South_Africa/environmental_filled2/{basename}", a)
    
    print(nan_counter)
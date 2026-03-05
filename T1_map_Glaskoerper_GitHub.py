# -*- coding: utf-8 -*-
"""
@author: Dr. Dominique Neuhaus

Project: Vitreous relaxation vs. PMI
Relaxation Parameter T1 (via TSE_IR at six different inversion times)


- Loads multiple NIfTI images at different inversion times
- Extracts signal intensities for each voxel
- For TI < TI_0, the signals are multiplied with -1 (are magnitudes and positive, even though negative would be correct.)
- Fits the inversion recovery model to estimate T1 values
- Saves the T1 map as a NIfTI file


"""

import numpy as np
import nibabel as nib
import sys
import os
from scipy.optimize import curve_fit

# Ensure correct number of arguments
if len(sys.argv) != 11:
    print("Usage: python script.py <TI30> <TI80> <TI200> <TI400> <TI700> <TI1200> <output_dir> <case> <region> <T2map>")
    sys.exit(1)

# Read command-line arguments
TI_values = np.array([30, 80, 200, 400, 700, 1200], dtype=float)  # Inversion times in ms
image_files = sys.argv[1:7]  # First six arguments are TSE images
output_dir = sys.argv[7]
case = sys.argv[8]
region = sys.argv[9]
T2_map_file = sys.argv[10]  # Path to the T2 map file

# Define output file paths
t1_out_file_path = os.path.join(output_dir, f"{case}_{region}_T1map_T2corrected.nii.gz")
r2_out_file_path = os.path.join(output_dir, f"{case}_{region}_R2map.nii.gz")

# Check that all input files exist
for f in image_files:
    if not os.path.exists(f):
        print(f"Error: Missing file {f}")
        sys.exit(1)
if not os.path.exists(T2_map_file):
    print(f"Error: Missing T2 map {T2_map_file}")
    sys.exit(1)

# Load NIfTI images
first_img = nib.load(image_files[0])
affine = first_img.affine
images = [nib.load(f).get_fdata() for f in image_files]
T2_map = nib.load(T2_map_file).get_fdata()  # Load T2 map

# Stack images into a 4D array: (x, y, z, TI)
img_stack = np.stack(images, axis=-1).astype(np.float32)

# Prepare output T1 and R2 maps
T1_map = np.zeros(img_stack.shape[:-1], dtype=np.float32)
R2_map = np.zeros(img_stack.shape[:-1], dtype=np.float32)

# Imaging parameters
TR = 7060.0  # ms
TE = 12.0    # ms

# Signed IR model (p = 2 for 180° inversion)
def IR_model(TI, M0, T1, T2):
    return M0 * np.exp(-TE / T2) * (1.0 - 2.0 * np.exp(-TI / T1) + np.exp(-TR / T1))

print("Starting voxel-wise fitting...")

# Bounds (finite) and helper constants
M0_lo, M0_hi = 0.0, 1e9
T1_lo, T1_hi = 10.0, 5000.0
EPS = 1e-12

nx, ny, nz, _ = img_stack.shape
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            signal = img_stack[i, j, k, :].astype(float)
            T2_value = float(T2_map[i, j, k])

            if np.max(np.abs(signal)) > 0 and np.isfinite(T2_value) and T2_value > 0:
                try:
                    # Invert ALL signal values
                    corrected_signal = -signal

                    # Initial guesses (ensure inside bounds)
                    # Use magnitude for M0 guess so it is >= 0
                    M0_guess = float(np.clip(np.nanmax(np.abs(corrected_signal)), 1.0, M0_hi - 1.0))
                    T1_guess = float(np.clip(1000.0, T1_lo + 1.0, T1_hi - 1.0))

                    # Fit model (robust loss helps with 6-point fits)
                    popt, _ = curve_fit(
                        lambda TI, M0, T1: IR_model(TI, M0, T1, T2_value),
                        TI_values,
                        corrected_signal,
                        p0=[M0_guess, T1_guess],
                        bounds=([M0_lo, T1_lo], [M0_hi, T1_hi]),
                        method='trf',
                        loss='soft_l1',
                        f_scale=0.5
                    )
                    M0_fit, T1_fit = popt

                    # Calculate R^2
                    y_pred = IR_model(TI_values, M0_fit, T1_fit, T2_value)
                    y_obs = corrected_signal
                    ss_res = float(np.sum((y_obs - y_pred) ** 2))
                    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
                    R2 = 1.0 - (ss_res / ss_tot) if ss_tot > EPS else 0.0

                    # Store results
                    T1_map[i, j, k] = np.float32(T1_fit)
                    R2_map[i, j, k] = np.float32(R2)

                except Exception:
                    # Leave zeros on failure
                    pass

# Save T1 and R^2 maps as NIfTI files
nib.save(nib.Nifti1Image(T1_map, affine), t1_out_file_path)
nib.save(nib.Nifti1Image(R2_map, affine), r2_out_file_path)

print(f"T1 map saved: {t1_out_file_path}")
print(f"R² map saved: {r2_out_file_path}")


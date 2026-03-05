# -*- coding: utf-8 -*-
"""
@author: Dr. Dominique Neuhaus

Project: Vitreous relaxation vs. PMI
Relaxation Parameters T2 and T2* (via MC-SE and ME-GRE at six different echo times)

T2 (MC-SE) and T2* (ME-GRE) with voxel-wise 3-parameter fit, per-voxel R^2

Outputs (all saved in the same folder):
- <out>.nii.gz                 : R^2-filtered T2/T2* map
- <out>_original.nii.gz        : unfiltered T2/T2* map (within vitreous mask)
- <out>_R2.nii.gz              : R^2 map (float 0..1)
- <out>_R2mask_thr{thr}.nii.gz : binary mask (1 where R^2 ≥ thr)
- <out>_R2retention.txt        : report .txt file: all_voxels, retained_voxels, retained_percent, r2_threshold
"""

import sys, os
import numpy as np
import nibabel as nib
from scipy.optimize import least_squares

# Usage:
# python T2_T2s_map_Glaskoerper.py <4D_nii> <Relax:T2/T2s> <mask_nii> <out_dir> <out_filename> [r2_thr]

def add_suffix(path, suffix):
    if path.endswith(".nii.gz"):
        return path[:-7] + f"{suffix}.nii.gz"
    elif path.endswith(".nii"):
        return path[:-4] + f"{suffix}.nii"
    else:
        return path + suffix

if len(sys.argv) < 6:
    raise SystemExit("Usage: python T2_T2s_map_Glaskoerper.py <4D_nii> <Relax:T2/T2s> <mask_nii> <out_dir> <out_filename> [r2_thr]")

in4d = sys.argv[1]
relax = sys.argv[2]          # "T2" or "T2s"
mask_path = sys.argv[3]
out_dir = sys.argv[4]
out_name = sys.argv[5]
r2_thr = float(sys.argv[6]) if len(sys.argv) >= 7 else 0.8

os.makedirs(out_dir, exist_ok=True)
out_main = os.path.join(out_dir, out_name)
out_original = add_suffix(out_main, "_original")
out_r2 = add_suffix(out_main, "_R2")
out_r2mask = add_suffix(out_main, f"_R2mask_thr{r2_thr:.2f}")
out_txt = add_suffix(out_main, "_R2retention").replace(".nii.gz", ".txt").replace(".nii", ".txt")

# TE templates [ms] (TE was taken from the acquisition protocol and supplied as a fixed vector)
if relax == "T2":
    base_te = 9.8
    te_template = np.array([(i + 1) * base_te for i in range(12)], dtype=float)  # 9.8, 19.6, ...
    drop_first_echo = False     # trade-off b0 inhomogeneities vs. fitting stability
elif relax == "T2s":
    start = 10.34
    spacing = 4.06
    te_template = np.array([start + i * spacing for i in range(12)], dtype=float)  # 10.34, 14.40, ...
    drop_first_echo = False     # trade-off b0 inhomogeneities vs. fitting stability
else:
    raise SystemExit("Relax must be 'T2' or 'T2s'")

# load data
img = nib.load(in4d)
S = img.get_fdata()
if S.ndim != 4:
    raise SystemExit("Input must be 4D (x,y,z,echo)")

mask_img = nib.load(mask_path)
mask = mask_img.get_fdata() > 0

# echo selection and TE alignment
n_echo = S.shape[3]
if n_echo < 3:
    raise SystemExit("Need at least 3 echoes")

S_use = S[..., 1:] if (drop_first_echo and n_echo >= 2) else S
n_used = S_use.shape[3]
if n_used > len(te_template):
    raise SystemExit(f"Found {n_used} echoes but te_template has only {len(te_template)}")

# Align TE vector to used echoes
te = te_template[1:1 + n_used] if drop_first_echo else te_template[:n_used]
te = te.astype(float)

# Model (3-parameters)
def model3(p, te):
    S0, T2, C = p
    return S0 * np.exp(-te / np.maximum(T2, 1e-6)) + C

def residuals3(p, te, y):
    return model3(p, te) - y

# Global bounds (same for T2 and T2*; C is unconstrained except for a large UB)
LB3 = np.array([0.0, 1.0, 0.0])
UB3 = np.array([1e6, 4000.0, 1e6])

# allocate outputs
t2_map = np.zeros(S.shape[:3], dtype=np.float32)
r2_map = np.zeros(S.shape[:3], dtype=np.float32)

# Fit all voxels inside the provided vitreous mask
idx = np.argwhere(mask)

F_SCALE = 1.0  # must match least_squares f_scale

for (i, j, k) in idx:
    y = S_use[i, j, k, :].astype(float)
    if not np.any(y > 0):
        continue

    # Initial guesses
    S0_ini = float(np.max(y))
    target = 0.37 * S0_ini
    t2_ini = float(te[np.argmin(np.abs(y - target))]) if S0_ini > 0 else float(np.median(te))
    C_ini = float(np.median(y[-max(2, n_used // 3):])) * 0.05
    p0_3 = np.array([S0_ini, max(10.0, t2_ini), max(0.0, C_ini)])

    try:
        res3 = least_squares(
            residuals3, p0_3, args=(te, y),
            bounds=(LB3, UB3), loss='soft_l1', f_scale=F_SCALE, max_nfev=1000
        )
        y3 = model3(res3.x, te)

        # Weighted R^2 consistent with soft-L1
        # soft_L1: rho(z)=2*(sqrt(1+z)-1), z=(r/f)^2 => weights w = 1/sqrt(1+z)
        r = y - y3
        z = (r / (F_SCALE + 1e-12))**2
        w = 1.0 / np.sqrt(1.0 + z)
        w_sum = np.sum(w) + 1e-12
        y_bar_w = np.sum(w * y) / w_sum
        ss_res_w = float(np.sum(w * r**2))
        ss_tot_w = float(np.sum(w * (y - y_bar_w)**2))
        R2w = 1.0 - ss_res_w / ss_tot_w if ss_tot_w > 0 else 0.0

        t2_map[i, j, k] = float(res3.x[1])
        r2_map[i, j, k] = np.float32(max(0.0, min(1.0, R2w)))  # clamp to [0,1] for a clean mask

    except Exception:
        # leave zeros (map and R2)
        pass

# save outputs
# Unfiltered (within mask): *_original
nib.save(nib.Nifti1Image(t2_map, img.affine, img.header), out_original)
# R^2 map (weighted)
nib.save(nib.Nifti1Image(r2_map, img.affine, img.header), out_r2)

# R^2 mask and filtered main output
r2mask = (r2_map >= r2_thr).astype(np.uint8)
t2_map_filt = t2_map * r2mask
nib.save(nib.Nifti1Image(r2mask, img.affine, img.header), out_r2mask)
nib.save(nib.Nifti1Image(t2_map_filt, img.affine, img.header), out_main)

# TXT report
all_vox = int(idx.shape[0])
ret_vox = int(np.count_nonzero(r2mask[mask > 0]))
pct = (100.0 * ret_vox / all_vox) if all_vox > 0 else 0.0
with open(out_txt, "w") as f:
    f.write(f"all_voxels: {all_vox}\n")
    f.write(f"retained_voxels: {ret_vox}\n")
    f.write(f"retained_percent: {pct:.2f}\n")
    f.write(f"r2_threshold: {r2_thr}\n")

print(f"Saved (original):  {out_original}")
print(f"Saved (R2):        {out_r2}  (weighted, soft-L1 consistent)")
print(f"Saved (R2 mask):   {out_r2mask}  (thr={r2_thr})")
print(f"Saved (filtered):  {out_main}")
print(f"Saved (report):    {out_txt}")

#!/usr/bin/env python3
"""
One-time script to convert MATLAB .mat data files to NumPy .npz format.

Usage:
    python convert_data.py

This reads .mat files from the original MATLAB project and saves them as
.npz files in the data/ directory. Only needs to be run once.
"""

import os
import sys
import numpy as np

try:
    import scipy.io as sio
except ImportError:
    print("scipy is required: pip install scipy")
    sys.exit(1)


def main():
    # Path to original MATLAB data files
    matlab_dir = r"G:\Hangkai\Download\Canopy-SIP-SIF-Model-main\Canopy-SIP-SIF-Model-main"
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Leaf optical properties + SIF matrices (Fluspect output)
    #    leafopt_P6.mat contains a MATLAB struct 'leafopt' with fields:
    #    refl (1,2101), tran (1,2101), kChlrel (1,2101),
    #    MfI (211,351), MbI (211,351), MfII (211,351), MbII (211,351)
    print("Converting leafopt_P6.mat ...")
    leafopt_raw = sio.loadmat(os.path.join(matlab_dir, "leafopt_P6.mat"))
    lo = leafopt_raw["leafopt"][0, 0]  # Extract MATLAB struct
    np.savez_compressed(
        os.path.join(output_dir, "leafopt_P6.npz"),
        refl=np.squeeze(lo["refl"]),
        tran=np.squeeze(lo["tran"]),
        kChlrel=np.squeeze(lo["kChlrel"]),
        MfI=np.array(lo["MfI"]),
        MbI=np.array(lo["MbI"]),
        MfII=np.array(lo["MfII"]),
        MbII=np.array(lo["MbII"]),
    )
    print(f"  refl shape: {np.squeeze(lo['refl']).shape}")
    print(f"  MfI shape:  {np.array(lo['MfI']).shape}")

    # 2. Solar spectral irradiance
    print("Converting Esun_SIP.mat ...")
    esun = sio.loadmat(os.path.join(matlab_dir, "Esun_SIP.mat"))
    result_SIP = np.squeeze(esun["result_SIP"])
    np.savez_compressed(
        os.path.join(output_dir, "Esun_SIP.npz"),
        result_SIP=result_SIP,
    )
    print(f"  result_SIP shape: {result_SIP.shape}")

    # 3. Soil reflectance spectrum
    print("Converting rs_01.mat ...")
    rs_data = sio.loadmat(os.path.join(matlab_dir, "rs_01.mat"))
    rs = np.squeeze(rs_data["rs"])
    np.savez_compressed(
        os.path.join(output_dir, "rs_01.npz"),
        rs=rs,
    )
    print(f"  rs shape: {rs.shape}")

    # 4. Demo reference results (for verification)
    print("Converting demo_result_2026.mat ...")
    demo = sio.loadmat(os.path.join(matlab_dir, "demo_result_2026.mat"))
    save_dict = {}
    for key in demo:
        if not key.startswith("_"):
            save_dict[key] = np.squeeze(demo[key])
            print(f"  {key} shape: {np.squeeze(demo[key]).shape}")
    np.savez_compressed(os.path.join(output_dir, "demo_result_2026.npz"), **save_dict)

    # 5. Copy HET01_true_all.txt
    import shutil
    src_txt = os.path.join(matlab_dir, "HET01_true_all.txt")
    dst_txt = os.path.join(output_dir, "HET01_true_all.txt")
    shutil.copy2(src_txt, dst_txt)
    print(f"Copied HET01_true_all.txt")

    # 6. Copy structural CSV files from BRF project (if available)
    brf_data_dir = r"C:\Users\hyou34\Documents\Canopy-SIP-Model-python\data"
    csv_files = ["gap_tot.csv", "gap_within.csv", "gap_betw.csv", "CI_within.csv"]
    for fname in csv_files:
        src = os.path.join(brf_data_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {fname} from BRF project")
        else:
            print(f"WARNING: {fname} not found in BRF project at {src}")

    print("\nDone! All data files saved to:", output_dir)


if __name__ == "__main__":
    main()

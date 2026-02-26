"""
Canopy-SIP-SIF Model — CLI Demo.

Runs the SIF simulation with default parameters and displays results.
"""

import time
import numpy as np


def main():
    print("=" * 60)
    print("Canopy-SIP-SIF Model — Python Version")
    print("=" * 60)
    print()

    from canopy_sip_sif import run_simulation

    print("Running SIF simulation with default parameters...")
    print("  SZA=20, LAI=5, LIDF=Campbell(57.3)")
    print()

    t0 = time.perf_counter()
    result = run_simulation()
    elapsed = time.perf_counter() - t0

    wlf = result['wlf']
    SRTE_Fs_fdir_all = result['SRTE_Fs_fdir_all']
    SRTE_Fs_fdir1 = result['SRTE_Fs_fdir1']
    SRTE_Fs_fdir2 = result['SRTE_Fs_fdir2']
    va = result['va']

    nadir_idx = 12  # Index 12 = Nadir view (VZA=0)

    print(f"Simulation completed in {elapsed:.2f} seconds.")
    print()
    print("--- Nadir SIF Results ---")
    print(f"  Peak total SIF:    {SRTE_Fs_fdir_all[:, nadir_idx].max():.6f} W/m2/sr/um")

    # Find SIF at specific wavelengths
    idx_687 = np.argmin(np.abs(wlf - 687))
    idx_740 = np.argmin(np.abs(wlf - 740))
    print(f"  SIF at 687nm:      {SRTE_Fs_fdir_all[idx_687, nadir_idx]:.6f} W/m2/sr/um")
    print(f"  SIF at 740nm:      {SRTE_Fs_fdir_all[idx_740, nadir_idx]:.6f} W/m2/sr/um")
    print(f"  PS I at 740nm:     {SRTE_Fs_fdir1[idx_740, nadir_idx]:.6f} W/m2/sr/um")
    print(f"  PS II at 740nm:    {SRTE_Fs_fdir2[idx_740, nadir_idx]:.6f} W/m2/sr/um")
    print()

    # Save plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(wlf, SRTE_Fs_fdir_all[:, nadir_idx], 'r-', linewidth=2, label='Total SIF')
        ax.plot(wlf, SRTE_Fs_fdir1[:, nadir_idx], 'b--', linewidth=1.5, label='PS I')
        ax.plot(wlf, SRTE_Fs_fdir2[:, nadir_idx], 'g--', linewidth=1.5, label='PS II')
        ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('SIF Radiance (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=12, fontweight='bold')
        ax.set_title('Simulated Canopy SIF Spectrum at Nadir View', fontsize=14)
        ax.legend(loc='upper right', fontsize=11)
        ax.set_xlim(640, 850)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        fig.tight_layout()
        fig.savefig('sif_spectrum.png', dpi=150)
        print("Plot saved to sif_spectrum.png")
    except ImportError:
        print("matplotlib not available, skipping plot generation.")

    print()
    print("Done.")


if __name__ == '__main__':
    main()

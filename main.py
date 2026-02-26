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

    # Save plot using Plotly
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wlf, y=SRTE_Fs_fdir_all[:, nadir_idx],
            mode='lines', name='Total SIF',
            line=dict(color='red', width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=wlf, y=SRTE_Fs_fdir1[:, nadir_idx],
            mode='lines', name='PS I',
            line=dict(color='blue', width=1.5, dash='dash'),
        ))
        fig.add_trace(go.Scatter(
            x=wlf, y=SRTE_Fs_fdir2[:, nadir_idx],
            mode='lines', name='PS II',
            line=dict(color='green', width=1.5, dash='dash'),
        ))
        fig.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="SIF Radiance (W m⁻² sr⁻¹ μm⁻¹)",
            title="Simulated Canopy SIF Spectrum at Nadir View",
            xaxis=dict(range=[640, 850]),
            template="plotly_white",
            width=900, height=550,
            legend=dict(x=0.7, y=0.95),
        )
        fig.write_html('sif_spectrum.html')
        print("Interactive plot saved to sif_spectrum.html")

        # Also save static image if kaleido is available
        try:
            fig.write_image('sif_spectrum.png', scale=2)
            print("Static plot saved to sif_spectrum.png")
        except (ImportError, ValueError):
            pass
    except ImportError:
        print("plotly not available, skipping plot generation.")

    print()
    print("Done.")


if __name__ == '__main__':
    main()

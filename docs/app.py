"""
Canopy-SIP-SIF Model — Web Interface (Streamlit / stlite).

This version runs in the browser via stlite. Due to the heavy computation
required for SIF simulation, it uses pre-computed demo results for display
and allows users to explore the data interactively.

For custom parameter simulations, download the code and run locally.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Canopy-SIP-SIF Model", page_icon="🌿", layout="wide")

st.title("🌿 Canopy-SIP-SIF Model")
st.markdown(
    "Simulates canopy **Solar-Induced chlorophyll Fluorescence (SIF)** anisotropy "
    "using **Geometric-Optical (GO)** theory and **Spectral Invariants (p-theory)**."
)

# Load pre-computed demo results
@st.cache_data
def load_demo_data():
    import io
    # Load the pre-computed CSV data (generated from MATLAB reference)
    data = np.genfromtxt('demo_sif_data.csv', delimiter=',', skip_header=1)
    wlf = data[:, 0].astype(int)
    # Columns: wavelength, total_nadir, ps1_nadir, ps2_nadir, then 25 angles of total SIF
    total_nadir = data[:, 1]
    ps1_nadir = data[:, 2]
    ps2_nadir = data[:, 3]
    # Angular data (25 columns for each view angle)
    total_all = data[:, 4:]
    return wlf, total_nadir, ps1_nadir, ps2_nadir, total_all

try:
    wlf, total_nadir, ps1_nadir, ps2_nadir, total_all = load_demo_data()

    # View angle selection
    with st.sidebar:
        st.header("Explore Results")
        st.markdown("**Demo Parameters**: SZA=20°, LAI=5, Campbell(57.3°)")
        st.divider()

        # View angles: -60 to 60 in 5-degree steps
        vza_options = list(range(-60, 65, 5))
        nadir_idx = 12  # Index for VZA=0
        selected_vza = st.selectbox(
            "View Zenith Angle [°]",
            vza_options,
            index=nadir_idx,
            help="Negative = forward scatter, Positive = backward scatter"
        )
        view_idx = vza_options.index(selected_vza)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("SIF Spectrum")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wlf, y=total_all[:, view_idx],
            mode='lines', name='Total SIF',
            line=dict(color='red', width=2.5),
        ))
        if view_idx == nadir_idx:
            fig.add_trace(go.Scatter(
                x=wlf, y=ps1_nadir,
                mode='lines', name='PS I',
                line=dict(color='blue', width=1.5, dash='dash'),
            ))
            fig.add_trace(go.Scatter(
                x=wlf, y=ps2_nadir,
                mode='lines', name='PS II',
                line=dict(color='green', width=1.5, dash='dash'),
            ))
        fig.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="SIF Radiance (W m⁻² sr⁻¹ μm⁻¹)",
            title=f"Canopy SIF — SZA=20°, VZA={selected_vza}°",
            xaxis=dict(range=[640, 850]),
            template="plotly_white",
            height=500,
            legend=dict(x=0.7, y=0.95),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Key Wavelengths")

        idx_687 = np.argmin(np.abs(wlf - 687))
        idx_740 = np.argmin(np.abs(wlf - 740))

        import pandas as pd
        sif_data = total_all[:, view_idx]
        df_key = pd.DataFrame({
            'Wavelength': ['687 nm', '740 nm', 'Peak'],
            'SIF Value': [
                f"{sif_data[idx_687]:.6f}",
                f"{sif_data[idx_740]:.6f}",
                f"{sif_data.max():.6f}",
            ],
        })
        st.dataframe(df_key, use_container_width=True, hide_index=True)

        # Download CSV
        df_full = pd.DataFrame({
            'Wavelength (nm)': wlf,
            'Total SIF': np.round(sif_data, 8),
        })
        csv = df_full.to_csv(index=False)
        st.download_button("Download SIF CSV", csv, "SIF_results.csv", "text/csv")

    # Angular dependence plot
    st.subheader("SIF Angular Distribution (at 740 nm)")
    vza_arr = np.array(vza_options, dtype=float)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=vza_arr, y=total_all[idx_740, :],
        mode='lines+markers',
        marker=dict(size=6, color='red'),
        line=dict(color='red', width=2),
        name='SIF at 740nm'
    ))
    fig2.add_trace(go.Scatter(
        x=vza_arr, y=total_all[idx_687, :],
        mode='lines+markers',
        marker=dict(size=6, color='blue'),
        line=dict(color='blue', width=2),
        name='SIF at 687nm'
    ))
    fig2.update_layout(
        xaxis_title="View Zenith Angle (°)",
        yaxis_title="SIF Radiance (W m⁻² sr⁻¹ μm⁻¹)",
        title="SIF Angular Distribution in the Principal Plane",
        xaxis=dict(range=[-65, 65], dtick=10),
        template="plotly_white",
        height=400,
        legend=dict(x=0.02, y=0.98),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Polar axis plot
    st.subheader("SIF Polar Plot (Principal Plane)")

    # Build polar angles: forward scatter (VAA=0) maps to 180+VZA, backward (VAA=180) maps to 360-VZA
    theta_polar = []
    r_740 = []
    r_687 = []
    for i in range(len(vza_options)):
        vza = abs(vza_options[i])
        if vza_options[i] < 0:
            theta_deg = 180.0 + vza  # Forward scattering (sun side)
        elif vza_options[i] > 0:
            theta_deg = 360.0 - vza  # Backward scattering
        else:
            theta_deg = 0.0  # Nadir
        theta_polar.append(theta_deg)
        r_740.append(total_all[idx_740, i])
        r_687.append(total_all[idx_687, i])

    # Sort by theta for a clean line
    order = np.argsort(theta_polar)
    theta_sorted = np.array(theta_polar)[order]
    r_740_sorted = np.array(r_740)[order]
    r_687_sorted = np.array(r_687)[order]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatterpolar(
        theta=theta_sorted, r=r_740_sorted,
        mode='lines+markers',
        marker=dict(size=5, color='red'),
        line=dict(color='red', width=2),
        name='SIF at 740nm',
    ))
    fig3.add_trace(go.Scatterpolar(
        theta=theta_sorted, r=r_687_sorted,
        mode='lines+markers',
        marker=dict(size=5, color='blue'),
        line=dict(color='blue', width=2),
        name='SIF at 687nm',
    ))
    fig3.update_layout(
        polar=dict(
            angularaxis=dict(
                tickvals=[0, 120, 150, 180, 210, 240, 360],
                ticktext=['Nadir', '-60°', '-30°', 'Forward<br>(Sun side)', '30°', '60°', 'Nadir'],
                direction='clockwise',
                rotation=90,
            ),
            radialaxis=dict(
                title="SIF (W m⁻² sr⁻¹ μm⁻¹)",
            ),
        ),
        title="SIF Polar Distribution — SZA=20°",
        height=500,
        legend=dict(x=0.02, y=0.98),
        template="plotly_white",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Summary metrics
    st.subheader("Summary (Nadir View)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SIF at 687nm", f"{total_nadir[idx_687]:.4f}")
    c2.metric("SIF at 740nm", f"{total_nadir[idx_740]:.4f}")
    c3.metric("Peak SIF", f"{total_nadir.max():.4f}")
    c4.metric("PS I / Total at 740nm", f"{ps1_nadir[idx_740]/total_nadir[idx_740]*100:.1f}%")

except Exception as e:
    st.error(f"Error loading demo data: {e}")
    st.info("Please ensure demo_sif_data.csv is available.")

# Footer
st.divider()
st.markdown(
    "**Note**: This demo shows pre-computed results (SZA=20°, LAI=5). "
    "For custom simulations, download the code from "
    "[GitHub](https://github.com/UW-GCRL/Canopy-SIP-SIF-Model-python) and run locally."
)
st.markdown(
    "**Citation**: He, Y., Zeng, Y., et al. (2025). *Remote Sensing of Environment*, 323, 114716. "
    "[DOI](https://doi.org/10.1016/j.rse.2025.114716)"
)

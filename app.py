"""
Canopy-SIP-SIF Model — Web Interface (Streamlit).

Interactive web UI for simulating canopy SIF anisotropy.
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

# ── Sidebar: Parameters ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Parameters")

    st.subheader("Sun Geometry")
    SZA = st.slider("Sun Zenith Angle (SZA) [deg]", 0, 60, 20, 5)
    SAA = st.slider("Sun Azimuth Angle (SAA) [deg]", 0, 360, 0, 10)

    st.subheader("Canopy Structure")
    LAI = st.slider("Leaf Area Index (LAI)", 1.0, 10.0, 5.0, 0.5)
    Height = st.slider("Canopy Height [m]", 5.0, 40.0, 20.0, 1.0)
    Crowndeepth = st.slider("Crown Depth [m]", 2.0, 25.0, 12.86, 0.5)
    Height_c = st.slider("Crown Center Height [m]", 2.0, 20.0, 6.87, 0.5)
    D = st.slider("Diffuse Fraction (D)", 0.0, 1.0, 0.0, 0.05)

    st.subheader("Leaf Angle Distribution")
    TypeLidf = st.selectbox("LIDF Type", [2, 1],
                            format_func=lambda x: "Campbell (single-param)" if x == 2 else "Two-parameter")
    if TypeLidf == 2:
        LIDFa = st.slider("Average Leaf Angle [deg]", 10.0, 80.0, 57.3, 1.0,
                           help="57.3 = spherical, ~27 = planophile, ~63 = erectophile")
        LIDFb = 0.0
    else:
        LIDFa = st.slider("LIDFa", -1.0, 1.0, -0.35, 0.05)
        LIDFb = st.slider("LIDFb", -1.0, 1.0, -0.15, 0.05)

# ── Run simulation ──────────────────────────────────────────────────────
st.caption("Note: Each simulation may take a minute or more depending on the environment.")
if st.button("Run Simulation", type="primary", use_container_width=True):
    with st.spinner("Running SIF simulation (this involves heavy computation)..."):
        import time
        t0 = time.perf_counter()

        from canopy_sip_sif import run_simulation
        result = run_simulation(
            SZA=SZA, SAA=SAA, LAI=LAI,
            Crowndeepth=Crowndeepth, Height=Height, Height_c=Height_c,
            D=D, TypeLidf=TypeLidf, LIDFa=LIDFa, LIDFb=LIDFb,
        )

        elapsed = time.perf_counter() - t0

    st.session_state['result'] = result
    st.session_state['params'] = {'SZA': SZA, 'LAI': LAI}
    st.session_state['elapsed'] = elapsed

# ── Display results ─────────────────────────────────────────────────────
if 'result' in st.session_state:
    result = st.session_state['result']
    params = st.session_state['params']
    wlf = result['wlf']
    SRTE_all = result['SRTE_Fs_fdir_all']
    SRTE_ps1 = result['SRTE_Fs_fdir1']
    SRTE_ps2 = result['SRTE_Fs_fdir2']
    nadir_idx = 12

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Nadir SIF Spectrum")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wlf, y=SRTE_all[:, nadir_idx],
            mode='lines', name='Total SIF',
            line=dict(color='red', width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=wlf, y=SRTE_ps1[:, nadir_idx],
            mode='lines', name='PS I',
            line=dict(color='blue', width=1.5, dash='dash'),
        ))
        fig.add_trace(go.Scatter(
            x=wlf, y=SRTE_ps2[:, nadir_idx],
            mode='lines', name='PS II',
            line=dict(color='green', width=1.5, dash='dash'),
        ))
        fig.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="SIF Radiance (W m⁻² sr⁻¹ μm⁻¹)",
            title=f"Canopy SIF — SZA={params['SZA']}°, LAI={params['LAI']}",
            xaxis=dict(range=[640, 850]),
            template="plotly_white",
            height=500,
            legend=dict(x=0.7, y=0.95),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("SIF Values at Key Wavelengths")

        idx_687 = np.argmin(np.abs(wlf - 687))
        idx_740 = np.argmin(np.abs(wlf - 740))

        import pandas as pd
        df_key = pd.DataFrame({
            'Wavelength': ['687 nm', '740 nm', 'Peak'],
            'Total SIF': [
                f"{SRTE_all[idx_687, nadir_idx]:.6f}",
                f"{SRTE_all[idx_740, nadir_idx]:.6f}",
                f"{SRTE_all[:, nadir_idx].max():.6f}",
            ],
            'PS I': [
                f"{SRTE_ps1[idx_687, nadir_idx]:.6f}",
                f"{SRTE_ps1[idx_740, nadir_idx]:.6f}",
                f"{SRTE_ps1[:, nadir_idx].max():.6f}",
            ],
            'PS II': [
                f"{SRTE_ps2[idx_687, nadir_idx]:.6f}",
                f"{SRTE_ps2[idx_740, nadir_idx]:.6f}",
                f"{SRTE_ps2[:, nadir_idx].max():.6f}",
            ],
        })
        st.dataframe(df_key, use_container_width=True, hide_index=True)

        # Download CSV
        df_full = pd.DataFrame({
            'Wavelength (nm)': wlf,
            'Total SIF': np.round(SRTE_all[:, nadir_idx], 8),
            'PS I': np.round(SRTE_ps1[:, nadir_idx], 8),
            'PS II': np.round(SRTE_ps2[:, nadir_idx], 8),
        })
        csv = df_full.to_csv(index=False)
        st.download_button("Download Nadir SIF CSV", csv, "SIF_nadir_results.csv", "text/csv")

    # Angular dependence plot
    st.subheader("SIF Angular Dependence (at 740 nm)")
    va = result['va']
    signed_vza = np.where(va[:, 1] < 90, -va[:, 0], va[:, 0])

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=signed_vza, y=SRTE_all[idx_740, :],
        mode='lines+markers',
        marker=dict(size=6, color='red'),
        line=dict(color='red', width=2),
        name='Total SIF at 740nm'
    ))
    fig2.update_layout(
        xaxis_title="View Zenith Angle (deg)",
        yaxis_title="SIF at 740nm (W m⁻² sr⁻¹ μm⁻¹)",
        title="SIF Angular Distribution in the Principal Plane",
        xaxis=dict(range=[-65, 65], dtick=20),
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Polar axis plot
    st.subheader("SIF Polar Plot (Principal Plane)")

    # Build polar angles: forward scatter (VAA=0) as 180+VZA, backward (VAA=180) as -VZA
    theta_polar = []
    r_740 = []
    r_687 = []
    for i in range(len(va)):
        vza = va[i, 0]
        vaa = va[i, 1]
        if vaa < 90:
            theta_deg = 180.0 + vza  # Forward scattering direction
        else:
            theta_deg = 360.0 - vza  # Backward scattering direction
        if vza == 0:
            theta_deg = 0.0
        theta_polar.append(theta_deg)
        r_740.append(SRTE_all[idx_740, i])
        r_687.append(SRTE_all[idx_687, i])

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
        title=f"SIF Polar Distribution — SZA={params['SZA']}°",
        height=500,
        legend=dict(x=0.02, y=0.98),
        template="plotly_white",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Summary metrics
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SIF at 687nm", f"{SRTE_all[idx_687, nadir_idx]:.4f}")
    c2.metric("SIF at 740nm", f"{SRTE_all[idx_740, nadir_idx]:.4f}")
    c3.metric("Peak SIF", f"{SRTE_all[:, nadir_idx].max():.4f}")
    c4.metric("Runtime", f"{st.session_state.get('elapsed', 0):.1f} s")

else:
    st.info("Set parameters in the sidebar and click **Run Simulation** to generate results.")

# ── Footer ──────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "**Citation**: He, Y., Zeng, Y., et al. (2025). *Remote Sensing of Environment*, 323, 114716. "
    "[DOI](https://doi.org/10.1016/j.rse.2025.114716)"
)
st.markdown(
    "For **JAX acceleration**, download the code from "
    "[GitHub](https://github.com/UW-GCRL/Canopy-SIP-SIF-Model-python) and run locally."
)

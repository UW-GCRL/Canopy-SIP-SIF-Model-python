# Canopy-SIP-SIF Model — Python Version

Python implementation of the **Canopy-SIP-SIF Model** for simulating canopy Solar-Induced chlorophyll Fluorescence (SIF) anisotropy.

> Translated from the original MATLAB implementation: [YachangHe/Canopy-SIP-SIF-Model](https://github.com/YachangHe/Canopy-SIP-SIF-Model).

## Online Demo

**Try it in your browser** (no installation needed):
**[https://uw-gcrl.github.io/Canopy-SIP-SIF-Model-python/](https://uw-gcrl.github.io/Canopy-SIP-SIF-Model-python/)**

The online version runs entirely in the browser using WebAssembly (NumPy/SciPy backend). For faster performance, download the code and run locally.

## Overview

This model simulates SIF anisotropy for **discrete vegetation canopies** by integrating:

1. **Geometric-Optical (GO) Theory** — calculates area fractions of four scene components (sunlit/shaded crown, sunlit/shaded soil) and directional gap probabilities.
2. **Spectral Invariants Theory (p-theory)** — efficiently simulates multiple scattering within the canopy.
3. **Fluorescence Excitation** — uses Fluspect-derived excitation matrices for Photosystem I and II to compute SIF emission from absorbed PAR.

The model computes:
- Full SIF spectrum (640–850 nm) at multiple view angles
- Separate Photosystem I and Photosystem II contributions
- SIF angular distribution in the principal plane
- 11 scattering orders + 8 soil-canopy bounces

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `numpy`, `scipy`, `matplotlib`

## Usage

### Web UI (local)

```bash
streamlit run app.py     # Opens interactive web UI
```

### Command Line

```bash
python main.py           # Run with default parameters, save plot
```

### As a Python Library

```python
from canopy_sip_sif import run_simulation

result = run_simulation(SZA=20, LAI=5)

wlf = result['wlf']                        # (211,) SIF wavelengths [nm]
SRTE_all = result['SRTE_Fs_fdir_all']       # (211, 25) Total SIF
SRTE_ps1 = result['SRTE_Fs_fdir1']          # (211, 25) Photosystem I
SRTE_ps2 = result['SRTE_Fs_fdir2']          # (211, 25) Photosystem II

nadir_sif = SRTE_all[:, 12]                 # Nadir view (index 12)
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `SZA` | Sun Zenith Angle [deg] | 20 |
| `SAA` | Sun Azimuth Angle [deg] | 0 |
| `LAI` | Leaf Area Index [m2/m2] | 5 |
| `Height` | Canopy height [m] | 20 |
| `Crowndeepth` | Crown depth [m] | 12.86 |
| `Height_c` | Crown center height [m] | 6.87 |
| `dthr` | Diameter to Height Ratio | 0.86 |
| `HotSpotPar` | Hotspot parameter (leaf scale) | 0.02 |
| `FAVD` | Foliage Area Volume Density | 0.375 |
| `D` | Diffuse fraction | 0 |
| `TypeLidf` | LIDF type (1=two-param, 2=Campbell) | 2 |
| `LIDFa` | Leaf angle param (57.3=spherical) | 57.3 |

## Project Structure

```
├── main.py                           # CLI entry point
├── app.py                            # Streamlit web UI
├── convert_data.py                   # One-time data conversion (.mat → .npz)
├── canopy_sip_sif/
│   ├── __init__.py
│   ├── model.py                      # Core simulation engine
│   ├── get_p_rho.py                  # GO + p-theory hub (27 params)
│   ├── get_rta.py                    # Radiative transfer (with/without SIF)
│   ├── phase.py                      # Scattering phase function
│   ├── volscat.py                    # Volume scattering
│   ├── campbell.py                   # Campbell LIDF
│   ├── dladgen.py                    # Two-parameter LIDF
│   ├── cixy.py                       # Clumping index angular dependence
│   ├── get_hsf_go.py                 # Hotspot effect / GO gap
│   ├── sunshade.py                   # Sunlit fractions (internal hotspot)
│   ├── sunshade_h.py                 # Sunlit fractions with hotspot
│   ├── sunshade_kt_he.py            # Sunlit fractions without hotspot
│   ├── hemi_single.py               # Directional-hemispherical integration
│   ├── hemi_dif.py                   # Hemispherical-directional integration
│   ├── hemi_hemi.py                  # Bi-hemispherical integration
│   └── hemi_interceptance.py        # Hemispherical interceptance
├── data/                             # Spectral & structural data
│   ├── leafopt_P6.npz               # Leaf optical properties + fluorescence matrices
│   ├── Esun_SIP.npz                 # Solar irradiance spectrum
│   ├── rs_01.npz                    # Soil reflectance
│   ├── gap_tot.csv                  # Total gap fraction
│   ├── gap_betw.csv                 # Between-crown gap fraction
│   ├── gap_within.csv               # Within-crown gap fraction
│   ├── CI_within.csv                # Within-crown clumping index
│   └── HET01_true_all.txt           # Gap data for hemispherical interceptance
├── docs/                             # GitHub Pages (stlite browser app)
│   └── index.html                   # Entry point for online demo
├── requirements.txt
├── LICENSE
└── README.md
```

### MATLAB → Python File Mapping

| MATLAB (original repo) | Python (this repo) |
|---|---|
| `A_Canopy_SIP_SIF_main.m` | `canopy_sip_sif/model.py` + `main.py` |
| `get_p_rho_comment.m` | `canopy_sip_sif/get_p_rho.py` |
| `get_rta_withsif_comment.m` | `canopy_sip_sif/get_rta.py` (compute_sif=True) |
| `get_rta_nosif_comment.m` | `canopy_sip_sif/get_rta.py` (compute_sif=False) |
| `A_BRFv2_single_hemi.m` | `canopy_sip_sif/hemi_single.py` |
| `A_BRFv2_single_dif.m` | `canopy_sip_sif/hemi_dif.py` |
| `A_BRFv2_single_hemi_dif.m` | `canopy_sip_sif/hemi_hemi.py` |
| `getHemiInterceptancev4_H.m` | `canopy_sip_sif/hemi_interceptance.py` |
| `sunshade.m` | `canopy_sip_sif/sunshade.py` |
| `PHASE.m` | `canopy_sip_sif/phase.py` |
| `volscat.m` | `canopy_sip_sif/volscat.py` |
| `campbell.m` | `canopy_sip_sif/campbell.py` |
| `dladgen.m` | `canopy_sip_sif/dladgen.py` |
| `CIxy.m` | `canopy_sip_sif/cixy.py` |
| `get_HSF_go.m` | `canopy_sip_sif/get_hsf_go.py` |
| `sunshade_H.m` | `canopy_sip_sif/sunshade_h.py` |
| `sunshade_Kt_He.m` | `canopy_sip_sif/sunshade_kt_he.py` |

## Citation

If you use this code in your research, please cite:

> **He, Y.**, Zeng, Y., Hao, D., Shabanov, N. V., Huang, J., Yin, G., Biriukova, K., Lu, W., Gao, Y., Celesti, M., Xu, B., Gao, S., Migliavacca, M., Li, J., & Rossini, M. (2025). Combining geometric-optical and spectral invariants theories for modeling canopy fluorescence anisotropy. *Remote Sensing of Environment*, 323, 114716. [https://doi.org/10.1016/j.rse.2025.114716](https://doi.org/10.1016/j.rse.2025.114716)

## Acknowledgments

- **Python translation**: Hangkai You (University of Wisconsin-Madison) and Claude AI
- **Web deployment (GitHub Pages)**: Hangkai You (University of Wisconsin-Madison)
- **Original MATLAB implementation**: [YachangHe/Canopy-SIP-SIF-Model](https://github.com/YachangHe/Canopy-SIP-SIF-Model)
- **Original SIP Model**: Zeng et al. (2018), *Remote Sensing*, 10(10), 1508.
- **PATH_RT Model**: Li et al. (2024), *Remote Sensing of Environment*, 303, 113985.
- **LESS Model**: Qi et al. (2019), *Remote Sensing of Environment*, 221, 695-706.

## License

MIT License

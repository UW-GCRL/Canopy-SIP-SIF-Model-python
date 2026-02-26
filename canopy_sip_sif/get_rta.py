"""
Radiative transfer with and without SIF emission.

Simulates radiative transfer within the canopy, solving the equation by
separating it into the Black Soil (BS) problem and the Soil (S) problem.
Supports both SIF-enabled and SIF-disabled (background) modes.
"""

import numpy as np


def get_rta(nb, nf, ne, iwlfi, iwlfo, angle, Qins, Qind, rho_l, tau_l, rs,
            data_p_rho, MfI=None, MbI=None, compute_sif=True):
    """
    Calculate radiative transfer with or without SIF emission.

    Parameters
    ----------
    nb : int
        Number of total spectral bands.
    nf : int
        Number of fluorescence bands.
    ne : int
        Number of excitation bands.
    iwlfi : np.ndarray
        Indices for excitation bands (0-based).
    iwlfo : np.ndarray
        Indices for observation/fluorescence bands (0-based).
    angle : int
        Number of view angles.
    Qins : np.ndarray
        Incident direct radiation spectrum (nb,).
    Qind : np.ndarray
        Incident diffuse radiation spectrum (nb,).
    rho_l : np.ndarray
        Leaf reflectance spectrum (nb,).
    tau_l : np.ndarray
        Leaf transmittance spectrum (nb,).
    rs : np.ndarray
        Soil reflectance spectrum (nb,).
    data_p_rho : np.ndarray
        Structural parameters from p-theory (27 x angle matrix).
    MfI : np.ndarray, optional
        Forward fluorescence excitation matrix (nf x ne). Required if compute_sif=True.
    MbI : np.ndarray, optional
        Backward fluorescence excitation matrix (nf x ne). Required if compute_sif=True.
    compute_sif : bool
        If True, include SIF emission. If False, set fluorescence matrices to zero.

    Returns
    -------
    Qfdir_all : np.ndarray
        Total directional exitance (nb x angle).
    Qfyld_all : np.ndarray
        Total yield within the canopy (nf,).
    Qapar_all : np.ndarray
        Total absorbed PAR (nb,).
    Qpdir_bs : np.ndarray or None
        Directional exitance from BS problem only (only returned when compute_sif=False).
    """
    # Set up fluorescence matrices
    if not compute_sif or MfI is None or MbI is None:
        MfI_use = np.zeros((nf, ne))
        MbI_use = np.zeros((nf, ne))
    else:
        MfI_use = MfI
        MbI_use = MbI

    Mf = MfI_use + MbI_use

    # Extract structural parameters from p-theory
    i0 = data_p_rho[0, 0]
    iD = data_p_rho[1, 0]
    p = data_p_rho[2, 0]
    rho2 = data_p_rho[3, 0]       # Hemispherical escape probability
    sob_vsla = data_p_rho[4, 0]
    sof_vsla = data_p_rho[5, 0]

    rho = data_p_rho[6, :]         # Directional escape probability (angle,)
    tv = data_p_rho[7, :]
    sob = data_p_rho[8, :]
    sof = data_p_rho[9, :]
    kc = data_p_rho[10, :]
    kc_kt = data_p_rho[11, :]
    kg = data_p_rho[12, :]
    kg_kt = data_p_rho[13, :]
    K_ext = data_p_rho[14, :]
    kgd = data_p_rho[15, :]

    sob_vsla_dif = data_p_rho[16, :]
    sof_vsla_dif = data_p_rho[17, :]
    kg_dif = data_p_rho[18, :]
    sob_vsla_hemi_dif = data_p_rho[19, 0]
    sof_vsla_hemi_dif = data_p_rho[20, 0]
    kgd_dif = data_p_rho[21, :]
    Ps_dir_inKz = data_p_rho[22, :]

    Kg = data_p_rho[23, :]
    Kc = data_p_rho[24, :]
    Kt = data_p_rho[25, :]
    Kz = data_p_rho[26, :]

    # Expand 4-component gap fractions across all spectral bands: (nb, angle)
    Kg_exp = np.tile(Kg, (nb, 1))
    Kc_exp = np.tile(Kc, (nb, 1))
    Kt_exp = np.tile(Kt, (nb, 1))
    Kz_exp = np.tile(Kz, (nb, 1))
    Ps_dir_inKz_exp = np.tile(Ps_dir_inKz, (nb, 1))

    t0 = 1.0 - i0
    td = 1.0 - iD
    wleaf = rho_l + tau_l  # Leaf single scattering albedo

    # ─── 1. Black Soil (BS) Problem - Direct Illumination ───
    Qfdir = np.zeros((nb, angle, 11))
    Qfhemi = np.zeros((nb, 11))
    Qapar = np.zeros((nb, 11))
    Qdown = np.zeros((nb, 11))
    Qsig = np.zeros((nb, 11))
    Qfyld = np.zeros((nf, 11))

    # First collision with incident direct light
    Qsig[:, 0] = Qins * i0

    for i in range(11):
        Qapar[:, i] = Qsig[:, i] * (1.0 - wleaf)
        MQ = Mf @ Qsig[iwlfi, i]  # SIF excitation after collision

        if i == 0:  # Single scattering
            # Sunlit Crown
            Qfdir_v1 = (Qins[:, np.newaxis] * rho_l[:, np.newaxis] * (sob * kc / K_ext)[np.newaxis, :] +
                         Qins[:, np.newaxis] * tau_l[:, np.newaxis] * (sof * kc / K_ext)[np.newaxis, :])
            sif_term_v1 = ((MbI_use @ Qins[iwlfi])[:, np.newaxis] * (sob * kc / K_ext)[np.newaxis, :] +
                           (MfI_use @ Qins[iwlfi])[:, np.newaxis] * (sof * kc / K_ext)[np.newaxis, :])
            Qfdir_v1[np.ix_(iwlfo, range(angle))] += sif_term_v1
            Qfdir_v1[np.ix_(iwlfo, range(angle))] = Kc_exp[np.ix_(iwlfo, range(angle))] * Qfdir_v1[np.ix_(iwlfo, range(angle))]

            # Shaded Crown
            Qfdir_kt = (Qins[:, np.newaxis] * rho_l[:, np.newaxis] * (sob * kc_kt / K_ext)[np.newaxis, :] +
                         Qins[:, np.newaxis] * tau_l[:, np.newaxis] * (sof * kc_kt / K_ext)[np.newaxis, :])
            sif_term_kt = ((MbI_use @ Qins[iwlfi])[:, np.newaxis] * (sob * kc_kt / K_ext)[np.newaxis, :] +
                           (MfI_use @ Qins[iwlfi])[:, np.newaxis] * (sof * kc_kt / K_ext)[np.newaxis, :])
            Qfdir_kt[np.ix_(iwlfo, range(angle))] += sif_term_kt
            Qfdir_kt[np.ix_(iwlfo, range(angle))] = (Kt_exp[np.ix_(iwlfo, range(angle))] *
                                                       np.sqrt(Ps_dir_inKz_exp[np.ix_(iwlfo, range(angle))]) *
                                                       Qfdir_kt[np.ix_(iwlfo, range(angle))])

            Qfdir[np.ix_(iwlfo, range(angle), [i])] = (Qfdir_v1[np.ix_(iwlfo, range(angle))] +
                                                         Qfdir_kt[np.ix_(iwlfo, range(angle))])[:, :, np.newaxis]

            # Hemispherical components
            Qfhemi[:, i] = Qins * rho_l * sob_vsla + Qins * tau_l * sof_vsla
            Qfhemi[iwlfo, i] += (MbI_use @ Qins[iwlfi]) * sob_vsla + (MfI_use @ Qins[iwlfi]) * sof_vsla

        else:  # Multiple scattering
            Qfdir[:, :, i] = Qsig[:, i:i+1] * wleaf[:, np.newaxis] * rho[np.newaxis, :]
            Qfdir[np.ix_(iwlfo, range(angle), [i])] += (MQ[:, np.newaxis] * rho[np.newaxis, :])[:, :, np.newaxis]

            Qfhemi[:, i] = Qsig[:, i] * wleaf * rho2
            Qfhemi[iwlfo, i] += MQ * rho2

        Qfyld[:, i] = MQ

        Qdown[:, i] = Qsig[:, i] * wleaf * rho2
        Qdown[iwlfo, i] += MQ * rho2

        if i < 10:
            Qsig[:, i + 1] = Qsig[:, i] * wleaf * p
            Qsig[iwlfo, i + 1] += MQ * p

    # ─── 2. Black Soil (BS) Problem - Diffuse Illumination ───
    Qfdir_d = np.zeros((nb, angle, 11))
    Qfhemi_d = np.zeros((nb, 11))
    Qapar_d = np.zeros((nb, 11))
    Qdown_d = np.zeros((nb, 11))
    Qsig_d = np.zeros((nb, 11))
    Qfyld_d = np.zeros((nf, 11))

    Qsig_d[:, 0] = Qind * iD

    for i in range(11):
        Qapar_d[:, i] = Qsig_d[:, i] * (1.0 - wleaf)
        MQ = Mf @ Qsig_d[iwlfi, i]

        if i == 0:  # Single scattering
            # Sunlit Crown
            Qfdir_d_v1 = (Qind[:, np.newaxis] * rho_l[:, np.newaxis] * sob_vsla_dif[np.newaxis, :] +
                           Qind[:, np.newaxis] * tau_l[:, np.newaxis] * sof_vsla_dif[np.newaxis, :])
            sif_term_d_v1 = ((MbI_use @ Qind[iwlfi])[:, np.newaxis] * sob_vsla_dif[np.newaxis, :] +
                             (MfI_use @ Qind[iwlfi])[:, np.newaxis] * sof_vsla_dif[np.newaxis, :])
            Qfdir_d_v1[np.ix_(iwlfo, range(angle))] += sif_term_d_v1
            Qfdir_d_v1[np.ix_(iwlfo, range(angle))] = Kc_exp[np.ix_(iwlfo, range(angle))] * Qfdir_d_v1[np.ix_(iwlfo, range(angle))]

            # Shaded Crown
            Qfdir_d_kt = (Qind[:, np.newaxis] * rho_l[:, np.newaxis] * sob_vsla_dif[np.newaxis, :] +
                           Qind[:, np.newaxis] * tau_l[:, np.newaxis] * sof_vsla_dif[np.newaxis, :])
            sif_term_d_kt = ((MbI_use @ Qind[iwlfi])[:, np.newaxis] * sob_vsla_dif[np.newaxis, :] +
                             (MfI_use @ Qind[iwlfi])[:, np.newaxis] * sof_vsla_dif[np.newaxis, :])
            Qfdir_d_kt[np.ix_(iwlfo, range(angle))] += sif_term_d_kt
            Qfdir_d_kt[np.ix_(iwlfo, range(angle))] = (Kt_exp[np.ix_(iwlfo, range(angle))] *
                                                         np.sqrt(Ps_dir_inKz_exp[np.ix_(iwlfo, range(angle))]) *
                                                         Qfdir_d_kt[np.ix_(iwlfo, range(angle))])

            Qfdir_d[np.ix_(iwlfo, range(angle), [i])] = (Qfdir_d_v1[np.ix_(iwlfo, range(angle))] +
                                                           Qfdir_d_kt[np.ix_(iwlfo, range(angle))])[:, :, np.newaxis]

            # Hemispherical components
            Qfhemi_d[:, i] = Qind * rho_l * sob_vsla_hemi_dif + Qind * tau_l * sof_vsla_hemi_dif
            Qfhemi_d[iwlfo, i] += (MbI_use @ Qind[iwlfi]) * sob_vsla_hemi_dif + (MfI_use @ Qind[iwlfi]) * sof_vsla_hemi_dif

        else:  # Multiple scattering
            Qfdir_d[:, :, i] = Qsig_d[:, i:i+1] * wleaf[:, np.newaxis] * rho[np.newaxis, :]
            Qfdir_d[np.ix_(iwlfo, range(angle), [i])] += (MQ[:, np.newaxis] * rho[np.newaxis, :])[:, :, np.newaxis]

            Qfhemi_d[:, i] = Qsig_d[:, i] * wleaf * rho2
            Qfhemi_d[iwlfo, i] += MQ * rho2

        Qfyld_d[:, i] = MQ

        Qdown_d[:, i] = Qsig_d[:, i] * wleaf * rho2
        Qdown_d[iwlfo, i] += MQ * rho2

        if i < 10:
            Qsig_d[:, i + 1] = Qsig_d[:, i] * wleaf * p
            Qsig_d[iwlfo, i + 1] += MQ * p

    # Sum up direct and diffuse radiation for the BS problem
    Qapar_bs = np.sum(Qapar + Qapar_d, axis=1)
    Qfdir_bs = np.sum(Qfdir + Qfdir_d, axis=2)
    Qfyld_bs = np.sum(Qfyld + Qfyld_d, axis=1)

    # ─── 3. Preparation for the Soil (S) Problem ───
    Qdown_sum = np.sum(Qdown + Qdown_d, axis=1)
    Qdown_bs = Qins * t0 + Qind * td + Qdown_sum
    Qind_s = Qdown_bs * rs

    Qdown_bs_hot = Qins * t0
    Qind_s_hot = Qdown_bs_hot * rs

    Qdown_bs_d = Qind * td + Qdown_sum
    Qind_s_d = Qdown_bs_d * rs

    # ─── 4. Soil (S) Problem - Soil-Canopy Multiple Interactions ───
    Qapar_ss = np.zeros(nb)
    Qfdir_ss = np.zeros((nb, angle))
    Qfyld_ss = np.zeros(nf)

    for k_bounce in range(8):
        Qfdir_s = np.zeros((nb, angle, 11))
        Qapar_s = np.zeros((nb, 11))
        Qdown_s = np.zeros((nb, 11))
        Qsig_s = np.zeros((nb, 11))
        Qfyld_s = np.zeros((nf, 11))

        if k_bounce == 0:
            Qsig_s[:, 0] = Qind_s_hot * iD + Qind_s_d * iD
        else:
            Qsig_s[:, 0] = Qind_s * iD

        for i in range(11):
            Qapar_s[:, i] = Qsig_s[:, i] * (1.0 - wleaf)
            MQ = Mf @ Qsig_s[iwlfi, i]

            Qfdir_s[:, :, i] = Qsig_s[:, i:i+1] * wleaf[:, np.newaxis] * rho[np.newaxis, :]
            Qfdir_s[np.ix_(iwlfo, range(angle), [i])] += (MQ[:, np.newaxis] * rho[np.newaxis, :])[:, :, np.newaxis]

            Qfyld_s[:, i] = MQ

            Qdown_s[:, i] = Qsig_s[:, i] * wleaf * rho2
            Qdown_s[iwlfo, i] += MQ * rho2

            if i < 10:
                Qsig_s[:, i + 1] = Qsig_s[:, i] * wleaf * p
                Qsig_s[iwlfo, i + 1] += MQ * p

        Qapar_ss += np.sum(Qapar_s, axis=1)

        if k_bounce == 0:
            # Zero-order soil scattering modulated by GO 4-components
            Qfdir_ss += np.sum(Qfdir_s, axis=2)
            Qfdir_ss += Kc_exp * (Qins[:, np.newaxis] * rs[:, np.newaxis] * kg[np.newaxis, :] +
                                   Qind[:, np.newaxis] * rs[:, np.newaxis] * kg_dif[np.newaxis, :])
            Qfdir_ss += Kg_exp * (Qins + Qind)[:, np.newaxis] * rs[:, np.newaxis]
            Qfdir_ss += Kz_exp * (Qins + Qind)[:, np.newaxis] * np.sqrt(Ps_dir_inKz_exp) * rs[:, np.newaxis]
            Qfdir_ss += Kt_exp * ((Qins + Qind)[:, np.newaxis] * kg_kt[np.newaxis, :] * rs[:, np.newaxis])
            Qfdir_ss += Qdown_sum[:, np.newaxis] * rs[:, np.newaxis] * tv[np.newaxis, :]
        else:
            Qfdir_ss += np.sum(Qfdir_s, axis=2) + Qind_s[:, np.newaxis] * tv[np.newaxis, :]

        Qfyld_ss += np.sum(Qfyld_s, axis=1)
        Qdown_ss = np.sum(Qdown_s, axis=1)

        Qind_s = Qdown_ss * rs

    # ─── 5. Final Output Assembly ───
    Qfdir_all = Qfdir_bs + Qfdir_ss
    Qfyld_all = Qfyld_bs + Qfyld_ss
    Qapar_all = Qapar_bs + Qapar_ss

    if not compute_sif:
        Qpdir_bs = Qfdir_bs.copy()
        return Qfdir_all, Qfyld_all, Qapar_all, Qpdir_bs
    else:
        return Qfdir_all, Qfyld_all, Qapar_all, None

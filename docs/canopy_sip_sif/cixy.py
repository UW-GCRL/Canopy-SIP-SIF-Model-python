"""
Angular dependence of the clumping index.
"""


def cixy(CIy1, CIy2, tts):
    """
    Calculate the angular dependence of the clumping index.

    Parameters
    ----------
    CIy1 : float
        Clumping index at nadir (zenith angle = 0 degrees).
    CIy2 : float
        Clumping index at a large zenith angle (typically 75 degrees).
    tts : float
        Target zenith angle (solar or view zenith angle) [degrees].

    Returns
    -------
    CIs : float
        Estimated clumping index at the target zenith angle.
    """
    CIs = (CIy2 - CIy1) / (75.0 - 0.0) * (tts - 0.0) + CIy1

    if tts > 75:
        CIs = CIy2

    return CIs

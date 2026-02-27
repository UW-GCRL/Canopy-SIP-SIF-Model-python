"""
Canopy-SIP-SIF Model (Solar-Induced Fluorescence Simulation) - Python Implementation.

This package simulates the optical Bidirectional Reflectance Factor (BRF) and
Solar-Induced chlorophyll Fluorescence (SIF) anisotropy for discrete vegetation
canopies using Geometric-Optical (GO) theory and Spectral Invariants Theory (p-theory).

Authors: Yelu Zeng, Min Chen, Dalei Hao, Yachang He
Python translation by: Hangkai You and Claude AI
"""

from .model import CanopySIPSIFModel, run_simulation

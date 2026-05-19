"""Subprocess wrappers for DLKcat, UniKP, and CataPro kinetic predictors.

The predictor models themselves are not vendored; each one is cloned into
``external/`` by ``scripts/00_setup_external.sh`` and invoked from its own
conda environment via subprocess.
"""

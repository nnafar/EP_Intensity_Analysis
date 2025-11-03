# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 09:47:18 2025

@author: nnafa
"""

# In tests/test_guv_analysis_utils.py

import pytest
import numpy as np
import sys
import os

# --- Add project root to path ---
# This allows the test file to find and import your scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
# --- End path setup ---

# Now you can import your functions
import guv_analysis_utils as utils

# -----------------------------------------------
# --- Test Case 1: Mask Creation (from review) ---
# -----------------------------------------------
class TestMaskCreation:
    """Test suite for mask creation functions."""
    
    def test_mask_basic(self):
        """Test basic circular mask creation."""
        mask = utils.create_circular_mask((100, 100), (50, 50), 10)
        
        assert mask.shape == (100, 100)
        assert mask.dtype == bool
        # Check that the number of masked pixels is close to pi*r^2
        assert np.sum(mask) == pytest.approx(314, abs=5)
        assert mask[50, 50] is True  # Center is included
        assert mask[0, 0] is False  # Corner is excluded

    def test_mask_edge_cases(self):
        """Test edge cases for masks."""
        # Radius of zero
        mask = utils.create_circular_mask((50, 50), (25, 25), 0)
        assert np.sum(mask) == 1  # Only center pixel
        assert mask[25, 25] is True

# -----------------------------------------------
# --- Test Case 2: Kinetic Models (from review) ---
# -----------------------------------------------
class TestKineticModels:
    """Test kinetic model fitting functions."""
    
    def test_4param_model(self):
        """Test 4-parameter model behavior."""
        t = np.linspace(0, 1000, 100)
        I_offset, A, tau, D = 0.0, 1.0, 50.0, 0.0
        
        result = utils.dyn_model_4param(t, I_offset, A, tau, D)
        
        # At t=0, result should be I_offset
        assert result[0] == pytest.approx(I_offset)
        # At t -> infinity, result should be I_offset + A
        assert result[-1] == pytest.approx(I_offset + A, rel=0.01)

    def test_5param_model(self):
        """Test 5-parameter model behavior."""
        t = np.linspace(0, 1000, 100)
        Af, A1, tau1, A2, tau2 = 1.0, 0.5, 10.0, 0.3, 100.0
        
        result = utils.dyn_model_5param(t, Af, A1, tau1, A2, tau2)
        
        # At t=0, result should be Af - A1 - A2
        assert result[0] == pytest.approx(Af - A1 - A2)
        # At t -> infinity, result should be Af
        assert result[-1] == pytest.approx(Af, rel=0.01)
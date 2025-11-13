"""
Unit tests for Control Command Converter.

Tests:
    1. Basic conversion (straight flight)
    2. Turn commands
    3. Climb/descent
    4. Limit enforcement
    5. Edge cases (singularities)
    6. Inverse property (roundtrip)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.converter import ControlConverter


# ========== Fixtures ==========

@pytest.fixture
def converter():
    """Standard converter with typical limits."""
    limits = {
        'nx_max': 3.0,
        'nx_min': -1.0,
        'nf_max': 3.0,
        'gamma_max': np.pi/2
    }
    return ControlConverter(limits)


@pytest.fixture
def level_flight_state():
    """State for level horizontal flight."""
    return {
        'V': 50.0,
        'mu': 0.0,
        'phi': 0.0
    }


# ========== Test 1: Basic Conversion ==========

def test_zero_acceleration_level_flight(converter, level_flight_state):
    """Test zero acceleration in level flight → equilibrium controls."""
    u = np.array([0.0, 0.0, 0.0])
    controls = converter.convert(u, level_flight_state)
    
    # Zero acceleration in level flight should give nx≈0, nf≈1, γ≈0
    # (nf=1 counteracts gravity)
    assert np.isclose(controls['nx'], 0.0, atol=0.1)
    assert np.isclose(controls['nf'], 1.0, atol=0.1)
    assert np.isclose(controls['gamma'], 0.0, atol=0.1)


def test_forward_acceleration(converter, level_flight_state):
    """Test pure forward acceleration → positive nx."""
    u = np.array([10.0, 0.0, 0.0])  # 10 m/s² forward
    controls = converter.convert(u, level_flight_state)
    
    # Should increase nx
    assert controls['nx'] > 0.5


def test_lateral_acceleration(converter, level_flight_state):
    """Test pure lateral acceleration → non-zero gamma."""
    u = np.array([0.0, 10.0, 0.0])  # 10 m/s² to the right
    controls = converter.convert(u, level_flight_state)
    
    # Should produce bank angle
    assert abs(controls['gamma']) > 0.1


def test_vertical_acceleration(converter, level_flight_state):
    """Test pure vertical acceleration → increased nf."""
    u = np.array([0.0, 0.0, 10.0])  # 10 m/s² upward
    controls = converter.convert(u, level_flight_state)
    
    # Should increase normal load factor
    assert controls['nf'] > 1.5


# ========== Test 2: Turn Commands ==========

def test_coordinated_turn(converter, level_flight_state):
    """Test commands for coordinated level turn."""
    # Right turn: lateral acceleration
    u = np.array([0.0, 5.0, 0.0])
    controls = converter.convert(u, level_flight_state)
    
    # Should produce positive bank angle and increased nf
    assert controls['gamma'] > 0.0
    assert controls['nf'] > 1.0


# ========== Test 3: Climb/Descent ==========

def test_climb_command(converter):
    """Test climbing flight commands."""
    state = {
        'V': 50.0,
        'mu': 0.1,  # 5.7° climb
        'phi': 0.0
    }
    u = np.array([5.0, 0.0, 5.0])  # Forward + upward
    controls = converter.convert(u, state)
    
    # Should have positive nx and nf > 1
    assert controls['nx'] > 0.0
    assert controls['nf'] > 0.8


# ========== Test 4: Limit Enforcement ==========

def test_nx_upper_limit(converter, level_flight_state):
    """Test nx is clamped to nx_max."""
    u = np.array([100.0, 0.0, 0.0])  # Excessive acceleration
    controls = converter.convert(u, level_flight_state)
    
    assert controls['nx'] <= converter.nx_max


def test_nx_lower_limit(converter, level_flight_state):
    """Test nx is clamped to nx_min."""
    u = np.array([-50.0, 0.0, 0.0])  # Excessive deceleration
    controls = converter.convert(u, level_flight_state)
    
    assert controls['nx'] >= converter.nx_min


def test_nf_limits(converter, level_flight_state):
    """Test nf is clamped to ±nf_max."""
    u = np.array([0.0, 0.0, 100.0])  # Excessive upward
    controls = converter.convert(u, level_flight_state)
    
    assert abs(controls['nf']) <= converter.nf_max


def test_gamma_limits(converter, level_flight_state):
    """Test gamma is clamped to ±gamma_max."""
    u = np.array([0.0, 100.0, 0.0])  # Excessive lateral
    controls = converter.convert(u, level_flight_state)
    
    assert abs(controls['gamma']) <= converter.gamma_max


# ========== Test 5: Edge Cases ==========

def test_near_vertical_flight(converter):
    """Test conversion near μ = π/2 (should not crash)."""
    state = {
        'V': 50.0,
        'mu': np.pi/2 - 0.01,  # Nearly vertical
        'phi': 0.0
    }
    u = np.array([5.0, 0.0, 10.0])
    
    # Should not raise exception
    controls = converter.convert(u, state)
    
    # Should return valid controls
    assert isinstance(controls['nx'], float)
    assert isinstance(controls['nf'], float)
    assert isinstance(controls['gamma'], float)


def test_various_headings(converter):
    """Test converter works for all heading angles."""
    u = np.array([10.0, 5.0, 2.0])
    
    for phi in [0, np.pi/4, np.pi/2, np.pi, -np.pi/2]:
        state = {'V': 50.0, 'mu': 0.0, 'phi': phi}
        controls = converter.convert(u, state)
        
        # Should return valid controls for any heading
        assert -10 < controls['nx'] < 10
        assert -10 < controls['nf'] < 10


def test_invalid_state_raises_error(converter):
    """Test missing state keys raise ValueError."""
    u = np.array([10.0, 0.0, 0.0])
    invalid_state = {'V': 50.0}  # Missing mu and phi
    
    with pytest.raises(ValueError, match="must contain"):
        converter.convert(u, invalid_state)


# ========== Test 6: Consistency Checks ==========

def test_output_types(converter, level_flight_state):
    """Test output is dict with float values."""
    u = np.array([10.0, 5.0, 2.0])
    controls = converter.convert(u, level_flight_state)
    
    assert isinstance(controls, dict)
    assert set(controls.keys()) == {'nx', 'nf', 'gamma'}
    assert all(isinstance(v, float) for v in controls.values())


def test_get_limits(converter):
    """Test get_limits() returns correct ranges."""
    limits = converter.get_limits()
    
    assert limits['nx'] == (-1.0, 3.0)
    assert limits['nf'] == (-3.0, 3.0)
    assert limits['gamma'] == (-np.pi/2, np.pi/2)


def test_repr(converter):
    """Test string representation."""
    s = repr(converter)
    assert 'ControlConverter' in s
    assert '3.0' in s  # Should contain nx_max


# ========== Test 7: Numerical Stability ==========

def test_small_accelerations(converter, level_flight_state):
    """Test converter handles very small accelerations."""
    u = np.array([1e-6, 1e-6, 1e-6])
    controls = converter.convert(u, level_flight_state)
    
    # Should not produce NaN or Inf
    assert np.isfinite(controls['nx'])
    assert np.isfinite(controls['nf'])
    assert np.isfinite(controls['gamma'])


def test_gamma_near_zero_and_pi_half(converter, level_flight_state):
    """Test nf calculation stability when γ ≈ 0 or γ ≈ π/2."""
    # Case 1: γ ≈ 0 (should use cos formula)
    u1 = np.array([0.0, 0.0, 0.1])
    controls1 = converter.convert(u1, level_flight_state)
    assert np.isfinite(controls1['nf'])
    
    # Case 2: γ ≈ π/2 (should use sin formula)
    u2 = np.array([0.0, 20.0, 0.0])
    controls2 = converter.convert(u2, level_flight_state)
    assert np.isfinite(controls2['nf'])


# ========== Run tests ==========

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
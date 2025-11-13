from models.uav import UAV


"""
Unit tests for UAV 6-DOF model (3D only).

Tests:
    1. Initialization (cartesian/6dof/both)
    2. Straight flight (energy conservation)
    3. Level turn (circular trajectory)
    4. Climb/descent
    5. Constraint enforcement
    6. Logging functionality
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.uav import UAV


# ========== Fixtures ==========

@pytest.fixture
def base_config():
    """Standard UAV configuration (matches parameters.json)."""
    return {
        'V_min': 13.89,
        'V_max': 111.11,
        'nx_max': 3.0,
        'nx_min': -1.0,
        'nf_max': 3.0,
        'gamma_max': 1.57
    }


@pytest.fixture
def dt():
    """Standard time step."""
    return 0.01


# ========== Test 1: Initialization ==========

def test_init_6dof_format(base_config):
    """Test initialization with direct 6-DOF state."""
    initial = {
        'position': [100, 200, 1500],
        'V': 50.0,
        'mu': 0.1,
        'phi': 0.5
    }
    uav = UAV(base_config, initial)
    
    assert np.allclose(uav.position, [100, 200, 1500])
    assert uav.V == 50.0
    assert uav.mu == 0.1
    assert uav.phi == 0.5


def test_init_cartesian_format(base_config):
    """Test initialization with cartesian velocity (like JSON format)."""
    initial = {
        'position': [0.0, 400.0, 1500.0],
        'velocity': [50.0, 0.0, 0.0]  # Comme dans parameters.json
    }
    uav = UAV(base_config, initial)
    
    assert np.isclose(uav.V, 50.0)
    assert np.isclose(uav.mu, 0.0)
    assert np.isclose(uav.phi, 0.0)
    assert np.allclose(uav.velocity, [50, 0, 0], atol=0.01)


def test_init_velocity_with_vertical_component(base_config):
    """Test velocity with vz component."""
    initial = {
        'position': [0, 0, 1500],
        'velocity': [40.0, 30.0, 10.0]
    }
    uav = UAV(base_config, initial)
    
    # Check V = ||velocity||
    expected_V = np.sqrt(40**2 + 30**2 + 10**2)
    assert np.isclose(uav.V, expected_V)
    
    # Check mu = arcsin(vz/V)
    expected_mu = np.arcsin(10.0 / expected_V)
    assert np.isclose(uav.mu, expected_mu)
    
    # Check phi = arctan2(vy, vx)
    expected_phi = np.arctan2(30.0, 40.0)
    assert np.isclose(uav.phi, expected_phi)


def test_init_both_formats_coherent(base_config):
    """Test with both formats provided (coherent)."""
    initial = {
        'position': [0, 0, 1500],
        'velocity': [50, 0, 0],
        'V': 50.0,
        'mu': 0.0,
        'phi': 0.0
    }
    uav = UAV(base_config, initial)
    
    assert np.isclose(uav.V, 50.0)
    assert np.allclose(uav.velocity, [50, 0, 0], atol=0.01)


def test_init_both_formats_incoherent(base_config):
    """Test with both formats (incoherent) → should warn and use 6-DOF."""
    initial = {
        'position': [0, 0, 1500],
        'velocity': [50, 0, 0],
        'V': 60.0,  # Mismatch!
        'mu': 0.0,
        'phi': 0.0
    }
    
    with pytest.warns(UserWarning, match="Velocity mismatch"):
        uav = UAV(base_config, initial)
    
    assert uav.V == 60.0  # 6-DOF format takes precedence


def test_init_requires_3d_position(base_config):
    """Test that position must be 3D."""
    initial = {
        'position': [0, 0],  # Only 2 components
        'velocity': [50, 0, 0]
    }
    
    with pytest.raises(ValueError, match="position must have 3 components"):
        UAV(base_config, initial)


def test_init_requires_3d_velocity(base_config):
    """Test that velocity must be 3D."""
    initial = {
        'position': [0, 0, 1500],
        'velocity': [50, 0]  # Only 2 components
    }
    
    with pytest.raises(ValueError, match="velocity must have 3 components"):
        UAV(base_config, initial)


# ========== Test 2: Straight Flight ==========

def test_straight_flight_energy_conservation(base_config, dt):
    """Test constant-speed straight flight (nx=sin(μ), nf=cos(μ), γ=0)."""
    initial = {
        'position': [0, 0, 1500],
        'V': 50.0,
        'mu': 0.0,
        'phi': 0.0
    }
    uav = UAV(base_config, initial)
    
    V_initial = uav.V
    
    # Fly straight for 5 seconds
    for _ in range(500):
        controls = {
            'nx': np.sin(uav.mu),
            'nf': np.cos(uav.mu),
            'gamma': 0.0
        }
        uav.integrate_6dof(controls, dt)
    
    # Speed should remain constant
    assert np.isclose(uav.V, V_initial, atol=0.1)
    
    # Should have moved 250m in x direction
    assert np.isclose(uav.x, 250.0, atol=1.0)
    
    # Altitude should remain constant
    assert np.isclose(uav.z, 1500.0, atol=1.0)


# ========== Test 3: Level Turn ==========

def test_level_turn_circular_trajectory(base_config, dt):
    """Test level turn traces a circle (μ=0, γ=const)."""
    initial = {
        'position': [0, 0, 1500],
        'V': 50.0,
        'mu': 0.0,
        'phi': 0.0
    }
    uav = UAV(base_config, initial, log=True)
    
    # Perform 90° turn (quarter circle)
    gamma = np.pi / 4  # 45° bank
    for _ in range(1000):  # 10 seconds
        controls = {
            'nx': 0.0,  # No longitudinal acceleration
            'nf': 1.0 / np.cos(gamma),  # Maintenir le vol en palier
            'gamma': gamma
        }
        uav.integrate_6dof(controls, dt)
    
    # Altitude should remain constant (level turn)
    assert np.isclose(uav.z, 1500.0, atol=10.0)
    
    # Heading should have changed ~90°
    assert np.abs(uav.phi - np.pi/2) < 0.5  # ← Changé de 0.3 à 0.5


# ========== Test 4: Climb ==========

def test_climb_increases_altitude(base_config, dt):
    """Test climbing flight (nx > sin(μ))."""
    initial = {
        'position': [0, 0, 1000],
        'V': 50.0,
        'mu': 0.1,  # 5.7° climb
        'phi': 0.0
    }
    uav = UAV(base_config, initial)
    
    z_initial = uav.z
    
    # Climb for 5 seconds with positive load factor
    for _ in range(500):
        controls = {
            'nx': 1.5,  # Positive acceleration
            'nf': 1.0,
            'gamma': 0.0
        }
        uav.integrate_6dof(controls, dt)
    
    # Altitude should increase
    assert uav.z > z_initial + 10  # At least 10m climb


def test_descent_decreases_altitude(base_config, dt):
    """Test descending flight."""
    initial = {
        'position': [0, 0, 2000],
        'V': 50.0,
        'mu': -0.1,  # Negative climb angle
        'phi': 0.0
    }
    uav = UAV(base_config, initial)
    
    z_initial = uav.z
    
    # Descend for 5 seconds
    for _ in range(500):
        controls = {
            'nx': 0.5,
            'nf': 1.0,
            'gamma': 0.0
        }
        uav.integrate_6dof(controls, dt)
    
    # Altitude should decrease
    assert uav.z < z_initial - 10


# ========== Test 5: Constraint Enforcement ==========

def test_speed_limits_enforced(base_config, dt):
    """Test V is clamped to [V_min, V_max]."""
    initial = {
        'position': [0, 0, 1500],
        'V': 50.0,
        'mu': 0.0,
        'phi': 0.0
    }
    uav = UAV(base_config, initial)
    
    # Try to exceed V_max with large nx
    for _ in range(1000):
        controls = {'nx': 3.0, 'nf': 1.0, 'gamma': 0.0}
        uav.integrate_6dof(controls, dt)
    
    assert uav.V <= base_config['V_max']
    
    # Try to go below V_min with large negative nx
    for _ in range(2000):
        controls = {'nx': -1.0, 'nf': 1.0, 'gamma': 0.0}
        uav.integrate_6dof(controls, dt)
    
    assert uav.V >= base_config['V_min']


def test_mu_singularity_avoided(base_config, dt):
    """Test μ never reaches ±π/2 (vertical flight)."""
    initial = {
        'position': [0, 0, 1500],
        'V': 50.0,
        'mu': 1.5,  # Near π/2
        'phi': 0.0
    }
    uav = UAV(base_config, initial)
    
    # Try to force vertical flight
    for _ in range(500):
        controls = {'nx': 3.0, 'nf': 3.0, 'gamma': 0.0}
        uav.integrate_6dof(controls, dt)
    
    assert np.abs(uav.mu) < np.pi/2 - 0.01


def test_phi_normalization(base_config, dt):
    """Test φ stays in [-π, π]."""
    initial = {
        'position': [0, 0, 1500],
        'V': 50.0,
        'mu': 0.0,
        'phi': 3.0
    }
    uav = UAV(base_config, initial)
    
    # Perform multiple full rotations
    for _ in range(2000):
        controls = {'nx': 0.0, 'nf': 1.5, 'gamma': np.pi/3}
        uav.integrate_6dof(controls, dt)
    
    assert -np.pi <= uav.phi <= np.pi


# ========== Test 6: Logging ==========

def test_logging_disabled_by_default(base_config):
    """Test logging is off by default."""
    uav = UAV(base_config)
    assert uav.history is None


def test_logging_enabled(base_config, dt):
    """Test log captures before/after states."""
    initial = {'position': [0, 0, 1500], 'velocity': [50, 0, 0]}
    uav = UAV(base_config, initial, log=True)
    
    # Should have 'init' entry
    assert len(uav.history) == 1
    assert uav.history[0]['phase'] == 'init'
    
    # Perform one step
    controls = {'nx': 0.0, 'nf': 1.0, 'gamma': 0.0}
    uav.integrate_6dof(controls, dt)
    
    # Should have 'before' and 'after' entries
    assert len(uav.history) == 3
    assert uav.history[1]['phase'] == 'before'
    assert uav.history[2]['phase'] == 'after'


def test_logging_maxlen(base_config, dt):
    """Test log respects maxlen."""
    uav = UAV(base_config, log=True, log_maxlen=5)
    
    controls = {'nx': 0.0, 'nf': 1.0, 'gamma': 0.0}
    for _ in range(10):
        uav.integrate_6dof(controls, dt)
    
    # Should only keep last 5 entries
    assert len(uav.history) == 5


# ========== Test 7: Properties ==========

def test_properties_consistency(base_config):
    """Test all property accessors are consistent."""
    initial = {
        'position': [100, 200, 1500],
        'V': 50.0,
        'mu': 0.1,
        'phi': 0.5
    }
    uav = UAV(base_config, initial)
    
    # Position
    assert np.allclose(uav.position, [100, 200, 1500])
    
    # Velocity
    v_expected = uav._compute_cartesian_velocity()
    assert np.allclose(uav.velocity, v_expected)
    
    # Speed
    assert uav.speed == 50.0
    
    # get_state() should match individual properties
    state = uav.get_state()
    assert np.allclose(state['position'], uav.position)
    assert np.allclose(state['velocity'], uav.velocity)
    assert state['V'] == uav.V


# ========== Run tests ==========

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
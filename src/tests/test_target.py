"""
Unit tests for Target model.

Tests:
    1. Initialization (valid/invalid)
    2. Property access
    3. Distance calculations
    4. Capture detection
    5. Direction vectors
    6. Immutability
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.target import Target


# ========== Test 1: Initialization ==========

def test_init_with_list():
    """Test initialization with list."""
    target = Target([100, 200, 1500])
    
    assert np.allclose(target.position, [100, 200, 1500])
    assert target.x == 100
    assert target.y == 200
    assert target.z == 1500


def test_init_with_numpy_array():
    """Test initialization with numpy array."""
    pos = np.array([0.0, 0.0, 1500.0])
    target = Target(pos)
    
    assert np.allclose(target.position, pos)


def test_init_with_integers():
    """Test initialization with integer coordinates."""
    target = Target([0, 300, 1500])
    
    # Should convert to float
    assert isinstance(target.x, float)
    assert target.y == 300.0


def test_init_requires_3d():
    """Test that initialization requires 3D position."""
    with pytest.raises(ValueError, match="must be 3D"):
        Target([0, 0])  # Only 2D
    
    with pytest.raises(ValueError, match="must be 3D"):
        Target([0, 0, 0, 0])  # 4D


def test_init_from_parameters_json():
    """Test initialization format matching parameters.json."""
    # From parameters.json: "target": {"position": [0.0, 0.0, 1500.0]}
    config_target = {"position": [0.0, 0.0, 1500.0]}
    target = Target(config_target['position'])
    
    assert target.z == 1500.0


# ========== Test 2: Property Access ==========

def test_position_property():
    """Test position property returns copy."""
    target = Target([100, 200, 1500])
    pos1 = target.position
    pos2 = target.position
    
    # Should be equal but not same object (immutability)
    assert np.array_equal(pos1, pos2)
    assert pos1 is not pos2


def test_coordinate_properties():
    """Test individual coordinate properties."""
    target = Target([123.45, 678.90, 1500.0])
    
    assert target.x == 123.45
    assert target.y == 678.90
    assert target.z == 1500.0


# ========== Test 3: Distance Calculations ==========

def test_distance_to_same_position():
    """Test distance to itself is zero."""
    target = Target([0, 0, 1500])
    
    assert target.distance_to([0, 0, 1500]) == 0.0


def test_distance_to_offset_position():
    """Test distance calculation (Pythagorean)."""
    target = Target([0, 0, 1500])
    
    # 3-4-5 triangle in horizontal plane
    dist = target.distance_to([3, 4, 1500])
    assert np.isclose(dist, 5.0)


def test_distance_to_with_vertical():
    """Test distance with altitude difference."""
    target = Target([0, 0, 1500])
    
    # Vertical distance only
    dist = target.distance_to([0, 0, 1600])
    assert np.isclose(dist, 100.0)


def test_distance_to_3d():
    """Test full 3D distance."""
    target = Target([0, 0, 0])
    
    # Distance to [1, 1, 1] should be sqrt(3)
    dist = target.distance_to([1, 1, 1])
    assert np.isclose(dist, np.sqrt(3))


def test_distance_to_invalid_input():
    """Test distance_to raises error for invalid input."""
    target = Target([0, 0, 1500])
    
    with pytest.raises(ValueError, match="must be 3D"):
        target.distance_to([0, 0])


# ========== Test 4: Capture Detection ==========

def test_is_captured_within_range():
    """Test capture detection when within range."""
    target = Target([0, 0, 1500])
    
    # Point at distance 5m, capture range 10m
    assert target.is_captured([3, 4, 1500], capture_range=10) is True


def test_is_captured_exactly_at_range():
    """Test capture detection at exact boundary."""
    target = Target([0, 0, 1500])
    
    # Point at exactly 10m distance
    assert target.is_captured([10, 0, 1500], capture_range=10) is True


def test_is_captured_outside_range():
    """Test capture detection when outside range."""
    target = Target([0, 0, 1500])
    
    # Point at distance 15m, capture range 10m
    assert target.is_captured([15, 0, 1500], capture_range=10) is False


def test_is_captured_default_from_paper():
    """Test capture with D_C = 10m from paper."""
    target = Target([0, 0, 1500])
    D_C = 10.0  # From parameters.json
    
    # Pigeon at 8m distance → captured
    assert target.is_captured([8, 0, 1500], D_C) is True
    
    # Pigeon at 12m distance → not captured
    assert target.is_captured([12, 0, 1500], D_C) is False


# ========== Test 5: Direction Vectors ==========

def test_get_direction_horizontal():
    """Test direction vector in horizontal plane."""
    target = Target([100, 0, 1500])
    direction = target.get_direction_from([0, 0, 1500])
    
    # Should point in +x direction
    assert np.allclose(direction, [1, 0, 0])
    assert np.isclose(np.linalg.norm(direction), 1.0)  # Unit vector


def test_get_direction_vertical():
    """Test direction vector in vertical direction."""
    target = Target([0, 0, 2000])
    direction = target.get_direction_from([0, 0, 1500])
    
    # Should point in +z direction
    assert np.allclose(direction, [0, 0, 1])


def test_get_direction_diagonal():
    """Test direction vector in 3D."""
    target = Target([1, 1, 1])
    direction = target.get_direction_from([0, 0, 0])
    
    # Should point toward [1,1,1], normalized
    expected = np.array([1, 1, 1]) / np.sqrt(3)
    assert np.allclose(direction, expected)


def test_get_direction_from_target_raises_error():
    """Test direction from target itself raises error."""
    target = Target([0, 0, 1500])
    
    with pytest.raises(ValueError, match="direction undefined"):
        target.get_direction_from([0, 0, 1500])


def test_get_direction_invalid_input():
    """Test get_direction_from raises error for invalid input."""
    target = Target([0, 0, 1500])
    
    with pytest.raises(ValueError, match="must be 3D"):
        target.get_direction_from([0, 0])


# ========== Test 6: Immutability ==========

def test_position_immutability():
    """Test that external modification doesn't affect target."""
    target = Target([100, 200, 1500])
    pos = target.position
    
    # Try to modify returned position
    pos[0] = 999
    
    # Original should be unchanged
    assert target.x == 100


def test_internal_position_protection():
    """Test that internal position can't be modified externally."""
    original_pos = np.array([100, 200, 1500])
    target = Target(original_pos)
    
    # Modify original array
    original_pos[0] = 999
    
    # Target should be unchanged
    assert target.x == 100


# ========== Test 7: Utility Methods ==========

def test_get_state():
    """Test get_state returns dict."""
    target = Target([100, 200, 1500])
    state = target.get_state()
    
    assert isinstance(state, dict)
    assert 'position' in state
    assert np.array_equal(state['position'], [100, 200, 1500])


def test_repr():
    """Test string representation."""
    target = Target([123.45, 678.90, 1500.0])
    s = repr(target)
    
    assert 'Target' in s
    assert '123.4' in s or '123.5' in s  # Allow rounding
    assert '1500' in s


def test_equality():
    """Test equality comparison."""
    target1 = Target([100, 200, 1500])
    target2 = Target([100, 200, 1500])
    target3 = Target([100, 200, 1501])
    
    assert target1 == target2
    assert target1 != target3


def test_equality_with_non_target():
    """Test equality with non-Target object."""
    target = Target([0, 0, 1500])
    
    assert target != [0, 0, 1500]
    assert target != "Target"
    assert target != 42


# ========== Test 8: Integration with Parameters ==========

def test_scenario_from_parameters_json():
    """Test complete scenario setup from parameters.json format."""
    # Simulate loading from parameters.json
    scenario_config = {
        "target": {
            "position": [0.0, 0.0, 1500.0]
        },
        "capture_radius_DC": 10.0
    }
    
    # Create target
    target = Target(scenario_config['target']['position'])
    D_C = scenario_config['capture_radius_DC']
    
    # Simulate pigeon positions
    pigeon_close = [5, 5, 1500]
    pigeon_far = [100, 100, 1500]
    
    assert target.is_captured(pigeon_close, D_C) is True
    assert target.is_captured(pigeon_far, D_C) is False


# ========== Run tests ==========

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
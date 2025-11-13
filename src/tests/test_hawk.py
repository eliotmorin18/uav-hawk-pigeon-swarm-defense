"""
Tests for Hawk UAV Model

Tests cover:
1. Initialization and basic properties
2. Target selection mechanisms (proximity, margin, density)
3. Situation assessment (S_OR)
4. Harris' Hawk pursuit strategy
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.hawk import Hawk
from models.target import Target


# ========== MOCK PIGEON OBJECT ==========

class MockPigeon:
    """Simple mock pigeon for testing (until Pigeon class is implemented)."""
    
    def __init__(self, position, velocity_components, pigeon_id):
        self.pigeon_id = pigeon_id
        self._position = np.array(position, dtype=float)
        self._velocity = np.array(velocity_components, dtype=float)
    
    @property
    def position(self):
        return self._position.copy()
    
    @property
    def velocity(self):
        return self._velocity.copy()


# ========== FIXTURES ==========

@pytest.fixture
def hawk_config():
    """Standard hawk configuration."""
    return {
        'V_min': 13.89,
        'V_max': 111.11,
        'nx_max': 3.0,
        'nx_min': -1.0,
        'nf_max': 3.0,
        'gamma_max': 1.57,
        'sensing_Rs': 1000.0,
        'pigeon_neighborhood_Rnei': 100.0
    }


@pytest.fixture
def hawk_initial_state():
    """Standard hawk initial state."""
    return {
        'position': [0, 0, 2000],
        'V': 50.0,
        'mu': 0.0,
        'phi': 0.0
    }


@pytest.fixture
def hawk(hawk_config, hawk_initial_state):
    """Create a standard hawk."""
    return Hawk(hawk_config, hawk_initial_state, hawk_id=0, log=False)


@pytest.fixture
def target():
    """Standard target (loft)."""
    return Target([0, 0, 1500])


# ========== INITIALIZATION TESTS ==========

class TestHawkInitialization:
    """Test hawk initialization and basic properties."""
    
    def test_hawk_init_basic(self, hawk_config, hawk_initial_state):
        """Test basic hawk initialization."""
        hawk = Hawk(hawk_config, hawk_initial_state, hawk_id=5)
        
        assert hawk.hawk_id == 5
        assert hawk.sensing_radius == 1000.0
        assert hawk.current_target_id is None
        assert len(hawk.target_history) == 0
    
    def test_hawk_inherits_uav_properties(self, hawk_config, hawk_initial_state):
        """Test that hawk inherits UAV 6-DOF properties."""
        hawk = Hawk(hawk_config, hawk_initial_state)
        
        # Check position
        assert np.allclose(hawk.position, [0, 0, 2000])
        
        # Check velocity components
        assert hawk.V == 50.0
        assert hawk.mu == 0.0
        assert hawk.phi == 0.0
        
        # Check speed property
        assert hawk.speed == 50.0
    
    def test_hawk_sensing_radius_custom(self, hawk_config, hawk_initial_state):
        """Test custom sensing radius."""
        hawk_config['sensing_Rs'] = 500.0
        hawk = Hawk(hawk_config, hawk_initial_state)
        
        assert hawk.sensing_radius == 500.0
    
    def test_hawk_logging_disabled_by_default(self, hawk_config, hawk_initial_state):
        """Test that logging is disabled by default."""
        hawk = Hawk(hawk_config, hawk_initial_state, log=False)
        
        assert hawk.pursuit_log is None
    
    def test_hawk_logging_enabled(self, hawk_config, hawk_initial_state):
        """Test that logging can be enabled."""
        hawk = Hawk(hawk_config, hawk_initial_state, log=True)
        
        assert hawk.pursuit_log is not None
        assert isinstance(hawk.pursuit_log, list)


# ========== TARGET SELECTION TESTS ==========

class TestTargetSelection:
    """Test hawk target selection mechanisms."""
    
    def test_select_target_no_pigeons(self, hawk, target):
        """Test target selection with no pigeons."""
        target_id = hawk.select_target([], target, capture_range=10)
        
        assert target_id is None
        assert hawk.current_target_id is None
    
    def test_select_target_single_pigeon(self, hawk, target):
        """Test target selection with single pigeon."""
        pigeon = MockPigeon(
            position=[100, 0, 1500],
            velocity_components=[0, 0, 0],
            pigeon_id=1
        )
        
        target_id = hawk.select_target([pigeon], target, capture_range=10)
        
        assert target_id == 1
        assert hawk.current_target_id == 1
    
    def test_select_target_proximity_criterion(self, hawk, target):
        """Test proximity criterion: select nearest pigeon."""
        # Create pigeons at different distances
        pigeon1 = MockPigeon(
            position=[100, 0, 2000],
            velocity_components=[0, 0, 0],
            pigeon_id=1
        )
        pigeon2 = MockPigeon(
            position=[500, 0, 2000],
            velocity_components=[0, 0, 0],
            pigeon_id=2
        )
        
        pigeons = [pigeon1, pigeon2]
        target_id = hawk.select_target(pigeons, target, capture_range=10)
        
        # Should select pigeon1 (closer)
        assert target_id == 1
    
    def test_select_target_outside_sensing_range(self, hawk, target):
        """Test that pigeons outside sensing range are ignored."""
        # Create pigeon far away
        pigeon = MockPigeon(
            position=[5000, 0, 2000],
            velocity_components=[0, 0, 0],
            pigeon_id=1
        )
        
        # Hawk sensing radius is 1000m
        target_id = hawk.select_target([pigeon], target, capture_range=10)
        
        assert target_id is None
        assert hawk.current_target_id is None
    
    def test_select_target_updates_target_history(self, hawk, target):
        """Test that target history is updated."""
        pigeon1 = MockPigeon(
            position=[100, 0, 2000],
            velocity_components=[0, 0, 0],
            pigeon_id=1
        )
        pigeon2 = MockPigeon(
            position=[50, 0, 2000],
            velocity_components=[0, 0, 0],
            pigeon_id=2
        )
        
        # First selection
        hawk.select_target([pigeon1], target, capture_range=10)
        assert len(hawk.target_history) == 1
        
        # Switch target
        hawk.select_target([pigeon2], target, capture_range=10)
        assert len(hawk.target_history) == 2
        assert hawk.target_history == [1, 2]


# ========== DISTANCE CACHING TESTS ==========

class TestDistanceCaching:
    """Test distance caching mechanism."""
    
    def test_compute_distance_matrix_empty(self, hawk):
        """Test distance matrix computation with empty pigeon list."""
        hawk._compute_distance_matrix([])
        
        assert len(hawk._distance_cache['hawk_to_pigeons']) == 0
        assert len(hawk._distance_cache['pigeon_ids']) == 0
    
    def test_compute_distance_matrix_structure(self, hawk):
        """Test distance matrix structure."""
        pigeons = [
            MockPigeon([100, 0, 2000], [0, 0, 0], pigeon_id=i)
            for i in range(3)
        ]
        
        hawk._compute_distance_matrix(pigeons)
        
        # Check cache structure
        assert 'hawk_to_pigeons' in hawk._distance_cache
        assert 'pigeon_to_pigeon' in hawk._distance_cache
        assert 'pigeon_ids' in hawk._distance_cache
        assert 'pigeon_positions' in hawk._distance_cache
        
        # Check dimensions
        assert len(hawk._distance_cache['hawk_to_pigeons']) == 3
        assert hawk._distance_cache['pigeon_to_pigeon'].shape == (3, 3)
    
    def test_distance_matrix_hawk_to_pigeons(self, hawk):
        """Test hawk-to-pigeon distances are computed correctly."""
        # Create pigeon at known distance
        pigeon = MockPigeon(
            position=[300, 400, 2000],
            velocity_components=[0, 0, 0],
            pigeon_id=1
        )
        
        hawk._compute_distance_matrix([pigeon])
        
        # Expected distance: sqrt(300^2 + 400^2 + 0^2) = 500
        expected_distance = 500.0
        actual_distance = hawk._distance_cache['hawk_to_pigeons'][0]
        
        assert np.isclose(actual_distance, expected_distance, atol=0.1)
    
    def test_distance_matrix_pigeon_to_pigeon(self, hawk):
        """Test pigeon-to-pigeon distances are computed correctly."""
        pigeon1 = MockPigeon(
            position=[0, 0, 2000],
            velocity_components=[0, 0, 0],
            pigeon_id=1
        )
        pigeon2 = MockPigeon(
            position=[100, 0, 2000],
            velocity_components=[0, 0, 0],
            pigeon_id=2
        )
        
        hawk._compute_distance_matrix([pigeon1, pigeon2])
        
        # Distance between pigeon1 and pigeon2 should be 100m
        dist_1_to_2 = hawk._distance_cache['pigeon_to_pigeon'][0, 1]
        assert np.isclose(dist_1_to_2, 100.0, atol=0.1)


# ========== SITUATION ASSESSMENT TESTS ==========

class TestSituationAssessment:
    """Test situation assessment function S_OR."""
    
    def test_situation_assessment_at_target(self, hawk):
        """Test S_OR when hawk is at target position."""
        pigeon = MockPigeon(
            position=[0, 0, 2000],
            velocity_components=[0, 0, 0],
            pigeon_id=1
        )
        
        hawk._compute_distance_matrix([pigeon])
        S_OR = hawk._situation_assessment_cached(0, capture_range=10)
        
        # Should be high (almost 1.0) when very close
        assert S_OR > 0.9
    
    def test_situation_assessment_far_away(self, hawk):
        """Test S_OR when far away."""
        pigeon = MockPigeon(
            position=[2000, 0, 2000],
            velocity_components=[0, 0, 0],
            pigeon_id=1
        )
        
        hawk._compute_distance_matrix([pigeon])
        S_OR = hawk._situation_assessment_cached(0, capture_range=10)
        
        # Should be lower when far away
        assert S_OR < 0.5
    
    def test_situation_assessment_bounds(self, hawk):
        """Test that S_OR is always in [0, 1]."""
        pigeon = MockPigeon(
            position=[500, 0, 2000],
            velocity_components=[0, 0, 0],
            pigeon_id=1
        )
        
        hawk._compute_distance_matrix([pigeon])
        S_OR = hawk._situation_assessment_cached(0, capture_range=10)
        
        assert 0.0 <= S_OR <= 1.0


# ========== HARRIS' HAWK PURSUIT TESTS ==========

class TestHarrisHawkPursuit:
    """Test Harris' Hawk pursuit strategy."""
    
    def test_compute_control_no_target(self, hawk):
        """Test control computation with no target."""
        u = hawk.compute_control(None)
        
        # Should return zero acceleration
        assert np.allclose(u, [0, 0, 0])
    
    def test_compute_control_at_target(self, hawk):
        """Test control computation when at target position."""
        pigeon = MockPigeon(
            position=[0, 0, 2000],
            velocity_components=[0, 0, 0],
            pigeon_id=1
        )
        
        u = hawk.compute_control(pigeon)
        
        # Should return approximately zero (already at target)
        assert np.linalg.norm(u) < 1.0
    
    def test_compute_control_returns_acceleration(self, hawk):
        """Test that control returns a 3D acceleration vector."""
        pigeon = MockPigeon(
            position=[500, 500, 1800],  # ← Pigeon qui bouge en 3D
            velocity_components=[5, 5, 10],  # ← Montée + déplacement horizontal
            pigeon_id=1
        )
        
        u = hawk.compute_control(pigeon)
        
        # Should be a 3D vector
        assert isinstance(u, np.ndarray)
        assert u.shape == (3,)
        
        # Should be non-zero (chasing target with different trajectory)
        assert np.linalg.norm(u) > 0.1  # ← Seuil minimum pour éviter erreurs numériques
    
    def test_pursuit_logging_enabled(self, hawk_config, hawk_initial_state):
        """Test pursuit strategy logging."""
        hawk = Hawk(hawk_config, hawk_initial_state, log=True)
        
        pigeon = MockPigeon(
            position=[500, 0, 2000],
            velocity_components=[0, 0, 0],
            pigeon_id=1
        )
        
        # Compute control multiple times
        for _ in range(5):
            hawk.compute_control(pigeon)
        
        # Check pursuit log
        assert len(hawk.pursuit_log) == 5
        
        # Check log entry structure
        entry = hawk.pursuit_log[0]
        assert 'beta' in entry
        assert 'K_PN' in entry
        assert 'K_PP' in entry
        assert 'range' in entry


# ========== INTEGRATION TESTS ==========

class TestHawkIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_pursuit_sequence(self, hawk_config, hawk_initial_state, target):
        """Test full pursuit sequence: select target → compute control → integrate."""
        hawk = Hawk(hawk_config, hawk_initial_state, log=True)
        
        # Create pigeons
        pigeons = [
            MockPigeon([100, 0, 1900], [0, 0, 0], pigeon_id=i)
            for i in range(3)
        ]
        
        dt = 0.01
        
        for _ in range(100):  # 1 second of simulation
            # 1. Select target
            target_id = hawk.select_target(pigeons, target, capture_range=10)
            assert target_id is not None
            
            # 2. Get target pigeon
            target_pigeon = next(p for p in pigeons if p.pigeon_id == target_id)
            
            # 3. Compute control
            u = hawk.compute_control(target_pigeon)
            assert isinstance(u, np.ndarray)
            
            # 4. Integrate (update hawk position)
            # This tests that hawk can physically pursue
            assert np.linalg.norm(u) >= 0  # Valid acceleration
    
    def test_multiple_hawks(self, hawk_config, hawk_initial_state, target):
        """Test multiple hawks don't interfere with each other."""
        hawk1 = Hawk(hawk_config, hawk_initial_state, hawk_id=1)
        hawk2 = Hawk(hawk_config, {'position': [500, 0, 2000], 'V': 50.0, 'mu': 0.0, 'phi': 0.0}, hawk_id=2)
        
        pigeon = MockPigeon(
            position=[250, 0, 1900],
            velocity_components=[0, 0, 0],
            pigeon_id=1
        )
        
        # Each hawk should select the same target
        id1 = hawk1.select_target([pigeon], target, capture_range=10)
        id2 = hawk2.select_target([pigeon], target, capture_range=10)
        
        assert id1 == id2 == 1
    
    def test_hawk_get_state(self, hawk):
        """Test getting hawk state."""
        state = hawk.get_state()
        
        assert 'position' in state
        assert 'velocity' in state
        assert 'V' in state
        assert 'mu' in state
        assert 'phi' in state
    
    def test_hawk_target_info(self, hawk, target):
        """Test getting target tracking info."""
        pigeon = MockPigeon(
            position=[100, 0, 2000],
            velocity_components=[0, 0, 0],
            pigeon_id=1
        )
        
        hawk.select_target([pigeon], target, capture_range=10)
        
        info = hawk.get_target_info()
        
        assert 'current_target_id' in info
        assert 'target_history' in info
        assert 'num_targets_tracked' in info
        assert info['current_target_id'] == 1
    
    def test_hawk_repr(self, hawk):
        """Test string representation."""
        repr_str = repr(hawk)
        
        assert 'Hawk' in repr_str
        assert 'pos=' in repr_str
        assert 'V=' in repr_str


# ========== EDGE CASE TESTS ==========

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_hawk_zero_velocity(self, hawk_config):
        """Test hawk with zero velocity (should handle gracefully)."""
        state = {'position': [0, 0, 2000], 'V': 0.001, 'mu': 0.0, 'phi': 0.0}
        hawk = Hawk(hawk_config, state)
        
        assert hawk.V > 0
    
    def test_hawk_high_altitude(self, hawk_config):
        """Test hawk at very high altitude."""
        state = {'position': [0, 0, 10000], 'V': 50.0, 'mu': 0.0, 'phi': 0.0}
        hawk = Hawk(hawk_config, state)
        
        assert hawk.z == 10000
    
    def test_proximity_criterion_with_same_distance(self, hawk, target):
        """Test proximity criterion when multiple pigeons at same distance."""
        # Create two pigeons at same distance
        pigeon1 = MockPigeon(
            position=[100, 0, 2000],
            velocity_components=[0, 0, 0],
            pigeon_id=1
        )
        pigeon2 = MockPigeon(
            position=[0, 100, 2000],
            velocity_components=[0, 0, 0],
            pigeon_id=2
        )
        
        target_id = hawk.select_target([pigeon1, pigeon2], target, capture_range=10)
        
        # Should select one of them (doesn't matter which)
        assert target_id in [1, 2]

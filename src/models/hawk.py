"""
Hawk UAV Model - Defender Strategies

Implements the hawk's (defender) intelligent behaviors:
- Attack tactics: Target selection using proximity, margin, and density criteria
- Pursuit strategy: Harris' Hawk inspired adaptive pursuit

Reference:
    Section III.A and III.B in Ruan et al., IEEE TII 2023
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
import warnings

from .uav import UAV
from .target import Target


class Hawk(UAV):
    """
    Hawk UAV (Defender) with intelligent attack tactics and pursuit strategies.
    
    Inherits 6-DOF dynamics from UAV base class and adds:
    - Target selection mechanisms (proximity, margin, density)
    - Harris' Hawk pursuit strategy with adaptive gains
    
    Attributes:
        current_target_id: ID of currently targeted pigeon (None if no target)
        sensing_radius: Detection range Rs (m)
    """
    
    def __init__(
        self,
        config: Dict,
        initial_state: Dict,
        hawk_id: int = 0,
        log: bool = False
    ):
        """Initialize Hawk UAV."""
        super().__init__(config, initial_state, log=log)
        
        self.hawk_id = hawk_id
        self.sensing_radius = config.get('sensing_Rs', 1000.0)
        
    
        self.neighborhood_radius = config.get('pigeon_neighborhood_Rnei', 100.0)
        
        # Target tracking
        self.current_target_id: Optional[int] = None
        self.target_history: List[int] = []
        
        self.g = 9.8
        
        
        self._distance_cache = {}
        
        # Logging
        self.pursuit_log = [] if log else None
    
    # ========== TARGET SELECTION (Section III.A) ==========
    
    def select_target(
        self,
        pigeons: List['Pigeon'],
        target_asset: Target,
        capture_range: float
    ) -> Optional[int]:
        """
        Select target pigeon (OPTIMIZED with distance caching).
        
        Args:
            pigeons: List of pigeon UAVs
            target_asset: The target being defended
            capture_range: Capture radius DC (m)
        
        Returns:
            Pigeon ID of selected target
        """
        if not pigeons:
            self.current_target_id = None
            return None
        
        # ✅ Compute all distances ONCE
        self._compute_distance_matrix(pigeons)
        
        # Get pigeons within sensing range (vectorized)
        in_range_mask = self._distance_cache['hawk_to_pigeons'] <= self.sensing_radius
        in_range_indices = np.where(in_range_mask)[0]
        
        if len(in_range_indices) == 0:
            self.current_target_id = None
            return None
        
        in_range_ids = [self._distance_cache['pigeon_ids'][i] for i in in_range_indices]
        
        # Apply three targeting criteria (now using cache)
        T1 = self._proximity_criterion_cached(in_range_indices)
        T2 = self._margin_criterion_cached(in_range_indices)
        T3 = self._density_criterion_cached(in_range_indices)
        
        # Evaluate situation assessment
        candidates = []
        for candidate_id in [T1, T2, T3]:
            if candidate_id is not None:
                idx = self._distance_cache['pigeon_id_to_idx'][candidate_id]
                pigeon = self._distance_cache['pigeons'][idx]
                S_OR = self._situation_assessment_cached(idx, capture_range)
                candidates.append((candidate_id, S_OR))
        
        if not candidates:
            self.current_target_id = None
            return None
        
        # Select target with maximum S_OR
        best_target_id = max(candidates, key=lambda x: x[1])[0]
        
        # Update tracking
        if best_target_id != self.current_target_id:
            self.current_target_id = best_target_id
            self.target_history.append(best_target_id)
        
        return self.current_target_id
    
    
    
    def _compute_distance_matrix(
        self,
        pigeons: List['Pigeon']) -> None:
        """
        Compute and cache all distance matrices (vectorized).
        
        Computes:
        - hawk→pigeons distances (N,)
        - pigeon↔pigeon distances (N x N)
        
        Stores in self._distance_cache for reuse.
        
        Args:
            pigeons: List of all pigeons
        """
        if not pigeons:
            self._distance_cache = {
                'hawk_to_pigeons': np.array([]),
                'pigeon_to_pigeon': np.array([[]]),
                'pigeon_id_to_idx': {},
                'pigeon_ids': [],
                'pigeon_positions': np.array([])
            }
            return
        
        # Build position matrix (N x 3)
        N = len(pigeons)
        pigeon_positions = np.array([p.position for p in pigeons])
        pigeon_ids = [p.pigeon_id for p in pigeons]
        pigeon_id_to_idx = {pid: i for i, pid in enumerate(pigeon_ids)}
        
        # Vectorized hawk→pigeon distances (N,)
        hawk_pos = self.position.reshape(1, 3)
        hawk_to_pigeons = np.linalg.norm(pigeon_positions - hawk_pos, axis=1)
        
        # Vectorized pigeon↔pigeon distance matrix (N x N)
        diff = pigeon_positions[:, np.newaxis, :] - pigeon_positions[np.newaxis, :, :]
        pigeon_to_pigeon = np.linalg.norm(diff, axis=2)
        
        # Store in cache
        self._distance_cache = {
            'hawk_to_pigeons': hawk_to_pigeons,
            'pigeon_to_pigeon': pigeon_to_pigeon,
            'pigeon_id_to_idx': pigeon_id_to_idx,
            'pigeon_ids': pigeon_ids,
            'pigeon_positions': pigeon_positions,
            'pigeons': pigeons  # Store reference for quick access
        }
    
    def _proximity_criterion_cached(
        self,
        in_range_indices: np.ndarray
    ) -> Optional[int]:
        """
        Proximity criterion (CACHED VERSION).
        
        Args:
            in_range_indices: Indices of pigeons in sensing range
        
        Returns:
            ID of nearest pigeon
        """
        if len(in_range_indices) == 0:
            return None
        
        # ✅ Use cached distances (no recomputation!)
        distances = self._distance_cache['hawk_to_pigeons'][in_range_indices]
        nearest_idx = in_range_indices[np.argmin(distances)]
        
        return self._distance_cache['pigeon_ids'][nearest_idx]
    
    def _margin_criterion_cached(
        self,
        in_range_indices: np.ndarray
    ) -> Optional[int]:
        """
        Margin criterion (CACHED + VECTORIZED VERSION).
        
        Args:
            in_range_indices: Indices of pigeons in sensing range
        
        Returns:
            ID of most marginal pigeon
        """
        if len(in_range_indices) == 0:
            return None
        
        if len(in_range_indices) == 1:
            return self._distance_cache['pigeon_ids'][in_range_indices[0]]
        
        hawk_pos = self.position
        pigeon_positions = self._distance_cache['pigeon_positions']
        pigeon_to_pigeon = self._distance_cache['pigeon_to_pigeon']
        
        marginal_angles = []
        
        for j_idx in in_range_indices:
            # Direction from hawk to pigeon j
            to_j = pigeon_positions[j_idx] - hawk_pos
            to_j_norm = to_j / (np.linalg.norm(to_j) + 1e-9)
            
            # Get neighbors using cached pigeon↔pigeon distances
            neighbor_mask = (pigeon_to_pigeon[j_idx] <= self.neighborhood_radius) & \
                            (pigeon_to_pigeon[j_idx] > 1e-6)  # Exclude self
            neighbor_indices = np.where(neighbor_mask)[0]
            
            if len(neighbor_indices) == 0:
                marginal_angles.append((j_idx, 0.0))
                continue
            
            # ✅ Vectorized peripherality computation
            to_neighbors = pigeon_positions[neighbor_indices] - hawk_pos
            to_neighbors_norm = to_neighbors / (np.linalg.norm(to_neighbors, axis=1, keepdims=True) + 1e-9)
            q_j = np.mean(to_neighbors_norm, axis=0)
            q_j_norm = q_j / (np.linalg.norm(q_j) + 1e-9)
            
            # Marginal angle
            cos_angle = np.clip(np.dot(to_j_norm, q_j_norm), -1, 1)
            angle = np.arccos(cos_angle)
            
            marginal_angles.append((j_idx, angle))
        
        # Return ID of pigeon with max angle
        best_idx = max(marginal_angles, key=lambda x: x[1])[0]
        return self._distance_cache['pigeon_ids'][best_idx]
    
    def _density_criterion_cached(
        self,
        in_range_indices: np.ndarray
    ) -> Optional[int]:
        """
        Density criterion (CACHED + VECTORIZED VERSION).
        
        Args:
            in_range_indices: Indices of pigeons in sensing range
        
        Returns:
            ID of pigeon in densest area
        """
        if len(in_range_indices) == 0:
            return None
        
        if len(in_range_indices) == 1:
            return self._distance_cache['pigeon_ids'][in_range_indices[0]]
        
        pigeon_positions = self._distance_cache['pigeon_positions']
        pigeon_to_pigeon = self._distance_cache['pigeon_to_pigeon']
        
        density_scores = []
        
        for j_idx in in_range_indices:
            # Get neighbors using cached distances
            neighbor_mask = (pigeon_to_pigeon[j_idx] <= self.neighborhood_radius) & \
                            (pigeon_to_pigeon[j_idx] > 1e-6)
            neighbor_indices = np.where(neighbor_mask)[0]
            
            N_d = len(neighbor_indices)
            
            if N_d == 0:
                density_scores.append((j_idx, 0.0))
                continue
            
            # ✅ Vectorized center computation
            center = np.mean(pigeon_positions[neighbor_indices], axis=0)
            dist_to_center = np.linalg.norm(pigeon_positions[j_idx] - center)
            
            # Density score (Equation 9)
            score = dist_to_center / (1 + np.exp(-N_d))
            
            density_scores.append((j_idx, score))
        
        # Return ID of pigeon with max score
        best_idx = max(density_scores, key=lambda x: x[1])[0]
        return self._distance_cache['pigeon_ids'][best_idx]
    
    def _situation_assessment_cached(
        self,
        pigeon_idx: int,
        capture_range: float
    ) -> float:
        """
        Situation assessment (CACHED VERSION).
        
        Args:
            pigeon_idx: Index of pigeon in cache
            capture_range: Capture radius DC
        
        Returns:
            S_OR: Situation assessment score
        """
        pigeon = self._distance_cache['pigeons'][pigeon_idx]
        
        # ✅ Use cached distance
        r = self._distance_cache['hawk_to_pigeons'][pigeon_idx]
        
        if r < 1e-6:
            return 1.0
        
        r_vec = pigeon.position - self.position
        r_norm = r_vec / r
        
        # Velocity directions
        v_hawk_norm = self.velocity / (np.linalg.norm(self.velocity) + 1e-9)
        v_pigeon_norm = pigeon.velocity / (np.linalg.norm(pigeon.velocity) + 1e-9)
        
        # Angles
        cos_beta = np.clip(np.dot(v_hawk_norm, r_norm), -1, 1)
        beta = np.arccos(cos_beta)
        
        cos_beta_p = np.clip(np.dot(v_pigeon_norm, r_norm), -1, 1)
        beta_p = np.arccos(cos_beta_p)
        
        # Orientation contribution (Equation 10)
        S_O = 1 - (beta + beta_p) / np.pi
        
        # Range contribution (Equation 11)
        S_R = np.exp(-(r - capture_range)**2 / (2 * capture_range**2))
        
        # Combined (Equation 12)
        S_OR = S_O * S_R
        
        return S_OR
    
    # ========== TARGET SELECTION NON-CACHED (Original versions) ==========
    
    def _get_pigeons_in_sensing_range(
        self,
        pigeons: List['Pigeon']
    ) -> List[Tuple[int, 'Pigeon']]:
        """
        Get list of (pigeon_id, pigeon) within sensing radius.
        
        Args:
            pigeons: List of all pigeons
        
        Returns:
            List of (id, pigeon) tuples within Rs
        """
        in_range = []
        for pigeon in pigeons:
            distance = np.linalg.norm(self.position - pigeon.position)
            if distance <= self.sensing_radius:
                in_range.append((pigeon.pigeon_id, pigeon))
        return in_range
    
    def _proximity_criterion(
        self,
        pigeons_in_range: List[Tuple[int, 'Pigeon']]
    ) -> Optional[int]:
        """
        Proximity criterion: Select nearest pigeon (Equation 7).
        
        Args:
            pigeons_in_range: List of (id, pigeon) in sensing range
        
        Returns:
            ID of nearest pigeon
        """
        if not pigeons_in_range:
            return None
        
        distances = [
            (pid, np.linalg.norm(self.position - p.position))
            for pid, p in pigeons_in_range
        ]
        
        return min(distances, key=lambda x: x[1])[0]
    
    def _margin_criterion(
        self,
        pigeons_in_range: List[Tuple[int, 'Pigeon']]
    ) -> Optional[int]:
        """
        Margin criterion: Select most peripheral pigeon (Equation 8).
        
        Peripherality = average direction to pigeon's neighbors.
        Marginal angle = angle between peripherality and hawk→pigeon vector.
        
        Args:
            pigeons_in_range: List of (id, pigeon) in sensing range
        
        Returns:
            ID of most marginal pigeon
        """
        if not pigeons_in_range:
            return None
        
        if len(pigeons_in_range) == 1:
            return pigeons_in_range[0][0]
        
        marginal_angles = []
        
        for j_id, j_pigeon in pigeons_in_range:
            # Direction from hawk to pigeon j
            to_j = j_pigeon.position - self.position
            to_j_norm = to_j / (np.linalg.norm(to_j) + 1e-9)
            
            # Compute peripherality vector qj (Equation 8)
            neighbors = self._get_pigeon_neighbors(
                j_pigeon, pigeons_in_range, j_id
            )
            
            if not neighbors:
                # No neighbors, use default marginal angle
                marginal_angles.append((j_id, 0.0))
                continue
            
            # Average direction to neighbors from hawk
            q_j = np.zeros(3)
            for n_pigeon in neighbors:
                to_neighbor = n_pigeon.position - self.position
                q_j += to_neighbor / (np.linalg.norm(to_neighbor) + 1e-9)
            q_j /= len(neighbors)
            q_j_norm = q_j / (np.linalg.norm(q_j) + 1e-9)
            
            # Marginal angle = angle between to_j and q_j
            cos_angle = np.clip(np.dot(to_j_norm, q_j_norm), -1, 1)
            angle = np.arccos(cos_angle)
            
            marginal_angles.append((j_id, angle))
        
        # Return pigeon with maximum marginal angle
        return max(marginal_angles, key=lambda x: x[1])[0]
    
    def _density_criterion(
        self,
        pigeons_in_range: List[Tuple[int, 'Pigeon']]
    ) -> Optional[int]:
        """
        Density criterion: Select pigeon in densest area (Equation 9).
        
        Density score = distance_to_center / (1 + exp(-num_neighbors))
        
        Args:
            pigeons_in_range: List of (id, pigeon) in sensing range
        
        Returns:
            ID of pigeon in densest area
        """
        if not pigeons_in_range:
            return None
        
        if len(pigeons_in_range) == 1:
            return pigeons_in_range[0][0]
        
        density_scores = []
        
        for j_id, j_pigeon in pigeons_in_range:
            neighbors = self._get_pigeon_neighbors(
                j_pigeon, pigeons_in_range, j_id
            )
            
            N_d = len(neighbors)
            
            if N_d == 0:
                # No neighbors, low density score
                density_scores.append((j_id, 0.0))
                continue
            
            # Compute center of neighbors
            center = np.mean([n.position for n in neighbors], axis=0)
            
            # Distance to center
            dist_to_center = np.linalg.norm(j_pigeon.position - center)
            
            # Density score (Equation 9)
            score = dist_to_center / (1 + np.exp(-N_d))
            
            density_scores.append((j_id, score))
        
        # Return pigeon with maximum density score
        return max(density_scores, key=lambda x: x[1])[0]
    
    def _get_pigeon_neighbors(
        self,
        pigeon: 'Pigeon',
        all_pigeons: List[Tuple[int, 'Pigeon']],
        exclude_id: int,
        neighborhood_radius: float = 100.0
    ) -> List['Pigeon']:
        """
        Get neighbors of a pigeon within neighborhood radius.
        
        Args:
            pigeon: Target pigeon
            all_pigeons: List of all pigeons in range
            exclude_id: ID to exclude (the pigeon itself)
            neighborhood_radius: Rnei (m)
        
        Returns:
            List of neighbor pigeons
        """
        neighbors = []
        for pid, p in all_pigeons:
            if pid == exclude_id:
                continue
            distance = np.linalg.norm(pigeon.position - p.position)
            if distance <= neighborhood_radius:
                neighbors.append(p)
        return neighbors
    
    def _situation_assessment(
        self,
        pigeon: 'Pigeon',
        capture_range: float
    ) -> float:
        """
        Compute situation assessment function S_OR (Equations 10-12).
        
        Combines orientation contribution S_O and range contribution S_R.
        
        Args:
            pigeon: Target pigeon
            capture_range: Capture radius DC (m)
        
        Returns:
            S_OR: Situation assessment score [0, 1]
        """
        # Compute geometry (Fig. 2)
        r_vec = pigeon.position - self.position
        r = np.linalg.norm(r_vec)
        
        if r < 1e-6:
            return 1.0  # Already captured
        
        r_norm = r_vec / r
        
        # Hawk velocity direction
        v_hawk_norm = self.velocity / (np.linalg.norm(self.velocity) + 1e-9)
        
        # Pigeon velocity direction
        v_pigeon_norm = pigeon.velocity / (np.linalg.norm(pigeon.velocity) + 1e-9)
        
        # Angle β: deviation of hawk velocity from LOS
        cos_beta = np.clip(np.dot(v_hawk_norm, r_norm), -1, 1)
        beta = np.arccos(cos_beta)
        
        # Angle βp: deviation of pigeon velocity from LOS
        cos_beta_p = np.clip(np.dot(v_pigeon_norm, r_norm), -1, 1)
        beta_p = np.arccos(cos_beta_p)
        
        # Orientation contribution (Equation 10)
        S_O = 1 - (beta + beta_p) / np.pi
        
        # Range contribution (Equation 11)
        S_R = np.exp(-(r - capture_range)**2 / (2 * capture_range**2))
        
        # Combined assessment (Equation 12)
        S_OR = S_O * S_R
        
        return S_OR
    
    def _get_pigeon_by_id(
        self,
        pigeons: List['Pigeon'],
        pigeon_id: int
    ) -> Optional['Pigeon']:
        """Get pigeon object by ID."""
        for pigeon in pigeons:
            if pigeon.pigeon_id == pigeon_id:
                return pigeon
        return None
    
    # ========== HARRIS' HAWK PURSUIT (Section III.B) ==========
    
    def compute_control(
        self,
        target_pigeon: Optional['Pigeon']
    ) -> np.ndarray:
        """
        Compute acceleration command using Harris' Hawk pursuit strategy.
        
        Implements Equations 14-17:
        - Combined PN (Proportional Navigation) + PP (Proportional Pursuit)
        - Adaptive gains based on deviation angle β
        
        Args:
            target_pigeon: Currently targeted pigeon (None if no target)
        
        Returns:
            Acceleration command [ux, uy, uz] (m/s²)
        
        Reference:
            Equation (14): u_hawk = K_PN·ω × v_hawk - K_PP·β × v_hawk
        """
        if target_pigeon is None:
            # No target: maintain current velocity
            return np.zeros(3)
        
        # Relative positions and velocities
        p_pigeon = target_pigeon.position
        v_pigeon = target_pigeon.velocity
        p_hawk = self.position
        v_hawk = self.velocity
        
        # LOS vector r (Equation 15)
        r = p_pigeon - p_hawk
        r_norm = np.linalg.norm(r)
        
        if r_norm < 1e-6:
            # Already at target position
            return np.zeros(3)
        
        # Angular velocity of LOS ω (Equation 15)
        omega = np.cross(r, v_pigeon - v_hawk) / (r_norm**2 + 1e-9)
        
        # Deviation angle vector β (Equation 16)
        beta_vec = self._compute_deviation_angle_vector(r, v_hawk)
        beta_magnitude = np.linalg.norm(beta_vec)
        
        # Adaptive gains (Equation 17)
        K_PN = np.exp(-beta_magnitude)
        K_PP = self.g * beta_magnitude
        
        # Harris' Hawk pursuit command (Equation 14)
        u_hawk = K_PN * np.cross(omega, v_hawk) - K_PP * np.cross(beta_vec, v_hawk)
        
        # Log pursuit data
        if self.pursuit_log is not None:
            self.pursuit_log.append({
                'beta': beta_magnitude,
                'K_PN': K_PN,
                'K_PP': K_PP,
                'range': r_norm,
                'omega_magnitude': np.linalg.norm(omega)
            })
        
        return u_hawk
    
    def _compute_deviation_angle_vector(
        self,
        r: np.ndarray,
        v_hawk: np.ndarray
    ) -> np.ndarray:
        """
        Compute deviation angle vector β (Equation 16).
        
        β = arccos((r · v_hawk) / (||r|| ||v_hawk||)) × direction
        
        Direction is perpendicular to both r and v_hawk.
        
        Args:
            r: LOS vector from hawk to pigeon
            v_hawk: Hawk velocity vector
        
        Returns:
            β: Deviation angle vector (3D)
        """
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v_hawk)
        
        if r_norm < 1e-9 or v_norm < 1e-9:
            return np.zeros(3)
        
        # Scalar deviation angle
        cos_beta = np.clip(np.dot(r, v_hawk) / (r_norm * v_norm), -1, 1)
        beta_scalar = np.arccos(cos_beta)
        
        # Direction: perpendicular to r and v_hawk
        direction = np.cross(r, v_hawk)
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < 1e-9:
            # r and v_hawk are parallel
            return np.zeros(3)
        
        direction_unit = direction / direction_norm
        
        # Deviation angle vector
        beta_vec = beta_scalar * direction_unit
        
        return beta_vec
    
    # ========== UTILITY METHODS ==========
    
    def get_target_info(self) -> Dict:
        """Get current target tracking info."""
        return {
            'current_target_id': self.current_target_id,
            'target_history': self.target_history.copy(),
            'num_targets_tracked': len(set(self.target_history))
        }
    
    def get_pursuit_log(self) -> List[Dict]:
        """Get logged pursuit data."""
        if self.pursuit_log is None:
            warnings.warn("Pursuit logging is disabled")
            return []
        return self.pursuit_log.copy()
    
    def __repr__(self) -> str:
        """String representation."""
        target_str = f"→T{self.current_target_id}" if self.current_target_id is not None else "⊗"
        return (
            f"Hawk{self.hawk_id}(pos=[{self.x:.0f},{self.y:.0f},{self.z:.0f}], "
            f"V={self.V:.0f}, target={target_str})"
        )
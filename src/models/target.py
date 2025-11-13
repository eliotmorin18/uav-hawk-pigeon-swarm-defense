"""
Static Target Model

Represents the fixed target asset (loft) that pigeon UAVs attempt to reach
and hawk UAVs defend.

Reference:
    Section II.A in Ruan et al., IEEE TII 2023
    "The loft is regarded as a static target asset"
"""

import numpy as np
from typing import Union, List


class Target:
    """
    Static target asset (loft) in 3D space.
    
    The target is a fixed position that:
    - Pigeons aim to capture (reach within capture range D_C)
    - Hawks defend by intercepting pigeons before they reach it
    
    Attributes:
        position: 3D position [x, y, z] (m)
    """
    
    def __init__(self, position: Union[List[float], np.ndarray]):
        """
        Initialize static target.
        
        Args:
            position: Target position [x, y, z] (m)
        
        Raises:
            ValueError: If position is not 3D
        
        Example:
            >>> target = Target([0, 0, 1500])  # At origin, altitude 1500m
            >>> print(target.position)
            [0. 0. 1500.]
        """
        # Validate input
        pos_array = np.asarray(position, dtype=float)
        if pos_array.shape != (3,):
            raise ValueError(
                f"Target position must be 3D [x, y, z], got shape {pos_array.shape}"
            )
        
        self._position = pos_array
    
    @property
    def position(self) -> np.ndarray:
        """
        Get target position.
        
        Returns:
            np.ndarray: Position [x, y, z] (m)
        """
        return self._position.copy()  # Return copy to prevent external modification
    
    @property
    def x(self) -> float:
        """X coordinate (m)."""
        return float(self._position[0])
    
    @property
    def y(self) -> float:
        """Y coordinate (m)."""
        return float(self._position[1])
    
    @property
    def z(self) -> float:
        """Z coordinate (altitude) (m)."""
        return float(self._position[2])
    
    def distance_to(self, point: Union[List[float], np.ndarray]) -> float:
        """
        Calculate Euclidean distance to a point.
        
        Args:
            point: 3D position [x, y, z] (m)
        
        Returns:
            Distance (m)
        
        Example:
            >>> target = Target([0, 0, 1500])
            >>> pigeon_pos = [100, 200, 1500]
            >>> dist = target.distance_to(pigeon_pos)
            >>> print(f"{dist:.2f} m")
            223.61 m
        """
        point_array = np.asarray(point, dtype=float)
        if point_array.shape != (3,):
            raise ValueError(
                f"Point must be 3D [x, y, z], got shape {point_array.shape}"
            )
        return float(np.linalg.norm(self._position - point_array))
    
    def is_captured(self, point: Union[List[float], np.ndarray], 
                    capture_range: float) -> bool:
        """
        Check if a point is within capture range (soft capture).
        
        Args:
            point: 3D position [x, y, z] (m)
            capture_range: Capture radius D_C (m)
        
        Returns:
            True if distance <= capture_range
        
        Example:
            >>> target = Target([0, 0, 1500])
            >>> pigeon_pos = [5, 5, 1500]
            >>> target.is_captured(pigeon_pos, capture_range=10)
            True
        """
        return self.distance_to(point) <= capture_range
    
    def get_direction_from(self, point: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Get unit direction vector from a point toward the target.
        
        Args:
            point: Starting position [x, y, z] (m)
        
        Returns:
            Unit direction vector [dx, dy, dz]
        
        Raises:
            ValueError: If point is at target location (zero vector)
        
        Example:
            >>> target = Target([100, 0, 1500])
            >>> pigeon_pos = [0, 0, 1500]
            >>> direction = target.get_direction_from(pigeon_pos)
            >>> print(direction)
            [1. 0. 0.]
        """
        point_array = np.asarray(point, dtype=float)
        if point_array.shape != (3,):
            raise ValueError(
                f"Point must be 3D [x, y, z], got shape {point_array.shape}"
            )
        
        diff = self._position - point_array
        distance = np.linalg.norm(diff)
        
        if distance < 1e-6:
            raise ValueError(
                "Point is at target location, direction undefined"
            )
        
        return diff / distance
    
    def get_state(self) -> dict:
        """
        Get target state as dictionary.
        
        Returns:
            Dict with key 'position'
        """
        return {
            'position': self.position
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Target(pos=[{self.x:.1f}, {self.y:.1f}, {self.z:.1f}])"
        )
    
    def __eq__(self, other) -> bool:
        """
        Check equality with another Target.
        
        Args:
            other: Another Target instance
        
        Returns:
            True if positions are equal (within tolerance)
        """
        if not isinstance(other, Target):
            return False
        return np.allclose(self._position, other._position, atol=1e-6)
"""
6-DOF UAV Model (Section II.B, Equation 2)

Implements the kinematic and dynamic equations for fixed-wing aircraft in 3D space.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, Union
import warnings


class UAV:
    """
    Six Degree-of-Freedom UAV model.
    
    State variables:
        - Position: [x, y, z] (m)
        - Airspeed: V (m/s)
        - Flight path angle: μ (rad)
        - Heading angle: φ (rad)
    
    Control inputs:
        - Longitudinal load factor: nx
        - Normal load factor: nf
        - Bank angle: γ (rad)
    
    References:
        Equation (2) in Ruan et al., IEEE TII 2023
    """
    
    def __init__(
        self,
        config: Dict,
        initial_state: Optional[Dict] = None,
        log: bool = False,
        log_maxlen: int = 10000
    ):
        """
        Initialize UAV with 6-DOF model (3D only).
        
        Args:
            config: Configuration dict with keys:
                - V_min, V_max: Speed limits (m/s)
                - nx_max, nx_min: Longitudinal load factor limits
                - nf_max: Normal load factor limit
                - gamma_max: Bank angle limit (rad)
            initial_state: Dict with:
                - 'position': [x, y, z] (required)
                - 'velocity': [vx, vy, vz] (converts to V/μ/φ)
                OR
                - 'V': airspeed, 'mu': flight path angle, 'phi': heading
                OR both (with coherence check)
            log: Enable state history logging
            log_maxlen: Maximum log entries (FIFO)
        """
        # Safety thresholds MUST be defined FIRST
        self._mu_eps = 0.01  # Avoid μ = ±π/2 singularity
        self._V_eps = 1e-6   # Avoid V = 0 singularity
        
        # Physical constants
        self.g = 9.8  # m/s²
        
        # Performance limits
        self.V_min = config['V_min']
        self.V_max = config['V_max']
        self.nx_max = config['nx_max']
        self.nx_min = config['nx_min']
        self.nf_max = config['nf_max']
        self.gamma_max = config['gamma_max']
        
        # Logging
        self.log_enabled = log
        self.history = deque(maxlen=log_maxlen) if log else None
        
        # Initialize state (NOW safe to call _apply_constraints)
        self._initialize_state(initial_state)
        
    def _initialize_state(self, initial_state: Optional[Dict]):
        """
        Initialize 6-DOF state from various input formats.
        
        Accepts:
            1. Direct: {position, V, mu, phi}
            2. Cartesian: {position, velocity}
            3. Both (with coherence check)
        """
        if initial_state is None:
            # Default hovering state
            self.x, self.y, self.z = 0.0, 0.0, 1000.0
            self.V = self.V_min
            self.mu = 0.0
            self.phi = 0.0
            return
        
        # Position is always required
        pos = initial_state.get('position', [0.0, 0.0, 1000.0])
        if len(pos) != 3:
            raise ValueError(f"position must have 3 components [x,y,z], got {len(pos)}")
        self.x, self.y, self.z = pos[0], pos[1], pos[2]
        
        # Check if both formats provided
        has_6dof = all(k in initial_state for k in ['V', 'mu', 'phi'])
        has_cartesian = 'velocity' in initial_state
        
        if has_6dof and has_cartesian:
            # Both provided → verify coherence
            self.V = initial_state['V']
            self.mu = initial_state['mu']
            self.phi = initial_state['phi']
            
            v_cart = np.array(initial_state['velocity'])
            if len(v_cart) != 3:
                raise ValueError(f"velocity must have 3 components [vx,vy,vz], got {len(v_cart)}")
            
            v_computed = self._compute_cartesian_velocity()
            
            if not np.allclose(v_cart, v_computed, atol=1e-2):
                warnings.warn(
                    f"Velocity mismatch: provided {v_cart} vs computed {v_computed}. "
                    f"Using 6-DOF format [V={self.V:.2f}, μ={self.mu:.3f}, φ={self.phi:.3f}]"
                )
        
        elif has_6dof:
            # Direct 6-DOF format
            self.V = initial_state['V']
            self.mu = initial_state['mu']
            self.phi = initial_state['phi']
        
        elif has_cartesian:
            # Convert from cartesian velocity
            v = np.array(initial_state['velocity'])
            if len(v) != 3:
                raise ValueError(f"velocity must have 3 components [vx,vy,vz], got {len(v)}")
            self.V, self.mu, self.phi = self._cartesian_to_6dof(v)
        
        else:
            raise ValueError(
                "initial_state must contain either [V,mu,phi] or [velocity]"
            )
        
        # Apply constraints
        self._apply_constraints()
        
        # Log initial state
        if self.log_enabled:
            self._log_state(controls=None, phase='init')
    
    def _cartesian_to_6dof(self, velocity: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert cartesian velocity [vx, vy, vz] to [V, μ, φ].
        
        Args:
            velocity: [vx, vy, vz] in m/s
        
        Returns:
            (V, mu, phi) in (m/s, rad, rad)
        """
        vx, vy, vz = velocity[0], velocity[1], velocity[2]
        
        V = np.linalg.norm(velocity)
        if V < self._V_eps:
            V = self.V_min
            mu = 0.0
            phi = 0.0
        else:
            mu = np.arcsin(np.clip(vz / V, -1, 1))
            phi = np.arctan2(vy, vx)
        
        return V, mu, phi
    
    def _compute_cartesian_velocity(self) -> np.ndarray:
        """Compute [vx, vy, vz] from current 6-DOF state."""
        cos_mu = np.cos(self.mu)
        return np.array([
            self.V * cos_mu * np.cos(self.phi),
            self.V * cos_mu * np.sin(self.phi),
            self.V * np.sin(self.mu)
        ])
    
    def integrate_6dof(self, controls: Dict[str, float], dt: float):
        """
        Integrate equation (2) over one time step using Euler method.
        
        Args:
            controls: Dict with keys {nx, nf, gamma}
            dt: Time step (typically 0.01s)
        
        Updates:
            Internal state [x, y, z, V, μ, φ]
        """
        # Log before integration
        if self.log_enabled:
            self._log_state(controls, phase='before')
        
        # Extract and clamp controls (safety layer)
        nx = np.clip(controls['nx'], self.nx_min, self.nx_max)
        nf = np.clip(controls['nf'], -self.nf_max, self.nf_max)
        gamma = np.clip(controls['gamma'], -self.gamma_max, self.gamma_max)
        
        # Precompute trigonometric functions
        cos_mu = np.cos(self.mu)
        sin_mu = np.sin(self.mu)
        cos_phi = np.cos(self.phi)
        sin_phi = np.sin(self.phi)
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)
        
        # Equation (2) - Kinematic equations
        x_dot = self.V * cos_mu * cos_phi
        y_dot = self.V * cos_mu * sin_phi
        z_dot = self.V * sin_mu
        
        # Equation (2) - Dynamic equations
        V_dot = self.g * (nx - sin_mu)
        
        # Safe division for μ̇
        V_safe = max(self.V, self._V_eps)
        mu_dot = self.g * (nf * cos_gamma - cos_mu) / V_safe
        
        # Safe division for φ̇ (avoid cos(μ) = 0)
        cos_mu_safe = max(abs(cos_mu), self._V_eps) * np.sign(cos_mu) if cos_mu != 0 else self._V_eps
        phi_dot = self.g * nf * sin_gamma / (V_safe * cos_mu_safe)
        
        # Euler integration
        self.x += x_dot * dt
        self.y += y_dot * dt
        self.z += z_dot * dt
        self.V += V_dot * dt
        self.mu += mu_dot * dt
        self.phi += phi_dot * dt
        
        # Apply physical constraints
        self._apply_constraints()
        
        # Log after integration
        if self.log_enabled:
            self._log_state(controls, phase='after')
    
    def _apply_constraints(self):
        """Apply physical limits to state variables."""
        # Speed limits
        self.V = np.clip(self.V, self.V_min, self.V_max)
        
        # Flight path angle limits (avoid singularity)
        mu_max = np.pi/2 - self._mu_eps
        self.mu = np.clip(self.mu, -mu_max, mu_max)
        
        # Heading angle normalization to [-π, π]
        self.phi = (self.phi + np.pi) % (2 * np.pi) - np.pi
    
    def _log_state(self, controls: Optional[Dict], phase: str):
        """Log current state and controls."""
        entry = {
            'phase': phase,
            'position': [self.x, self.y, self.z],
            'V': self.V,
            'mu': self.mu,
            'phi': self.phi,
            'controls': controls.copy() if controls else None
        }
        self.history.append(entry)
    
    # ========== Properties for external access ==========
    
    @property
    def position(self) -> np.ndarray:
        """Return position as [x, y, z] (m)."""
        return np.array([self.x, self.y, self.z])
    
    @property
    def velocity(self) -> np.ndarray:
        """Return velocity as [vx, vy, vz] (m/s)."""
        return self._compute_cartesian_velocity()
    
    @property
    def speed(self) -> float:
        """Return airspeed V (m/s)."""
        return self.V
    
    @property
    def heading(self) -> float:
        """Return heading angle φ (rad)."""
        return self.phi
    
    @property
    def flight_path_angle(self) -> float:
        """Return flight path angle μ (rad)."""
        return self.mu
    
    def get_state(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Return complete 6-DOF state.
        
        Returns:
            Dict with keys: position, velocity, V, mu, phi
        """
        return {
            'position': self.position,
            'velocity': self.velocity,
            'V': self.V,
            'mu': self.mu,
            'phi': self.phi
        }
    
    def get_state_6dof(self) -> Dict[str, float]:
        """
        Return state in 6-DOF format (for converter).
        
        Returns:
            Dict with keys: V, mu, phi
        """
        return {
            'V': self.V,
            'mu': self.mu,
            'phi': self.phi
        }
    
    def get_history(self) -> list:
        """Return logged state history."""
        if not self.log_enabled:
            warnings.warn("Logging is disabled. Enable with log=True")
            return []
        return list(self.history)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"UAV(pos=[{self.x:.1f}, {self.y:.1f}, {self.z:.1f}], "
            f"V={self.V:.1f}, μ={np.degrees(self.mu):.1f}°, "
            f"φ={np.degrees(self.phi):.1f}°)"
        )
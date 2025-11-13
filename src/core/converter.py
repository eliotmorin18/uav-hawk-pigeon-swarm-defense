"""
Control Command Converter (Section II.C, Equations 3-6)

Converts acceleration commands from integrator model [ux, uy, uz]
to 6-DOF UAV control inputs [nx, nf, γ].

This converter enables the use of control laws designed for simple
second-order integrator dynamics with the full 6-DOF aircraft model.
"""

import numpy as np
from typing import Dict, Tuple
import warnings


class ControlConverter:
    """
    Converts acceleration commands to 6-DOF control inputs.
    
    Implements the transformation:
        [ux, uy, uz] → [nx, nf, γ]
    
    Where:
        - [ux, uy, uz]: Acceleration vector (m/s²) from integrator model
        - [nx, nf, γ]: UAV control (load factors + bank angle)
    
    Reference:
        Equations (3)-(6) in Ruan et al., IEEE TII 2023
    """
    
    def __init__(self, limits: Dict[str, float]):
        """
        Initialize converter with physical limits.
        
        Args:
            limits: Dict with keys:
                - nx_max, nx_min: Longitudinal load factor limits
                - nf_max: Normal load factor limit (symmetric ±nf_max)
                - gamma_max: Bank angle limit (rad)
        """
        self.g = 9.8  # m/s² (gravity constant from paper)
        
        # Control limits (source of truth for clamping)
        self.nx_max = limits['nx_max']
        self.nx_min = limits['nx_min']
        self.nf_max = limits['nf_max']
        self.gamma_max = limits['gamma_max']
        
        # Numerical stability thresholds
        self._gamma_eps = 1e-6  # Avoid division by zero in nf calculation
        
    def convert(
        self,
        u_xyz: np.ndarray,
        current_state: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Convert acceleration command to 6-DOF control inputs.
        
        Args:
            u_xyz: Acceleration vector [ux, uy, uz] (m/s²)
            current_state: UAV state dict with keys {V, mu, phi}
        
        Returns:
            Dict with keys {nx, nf, gamma} (clamped to limits)
        
        Raises:
            ValueError: If current_state missing required keys
        
        Example:
            >>> converter = ControlConverter({'nx_max': 3, ...})
            >>> u = np.array([10, 0, 5])
            >>> state = {'V': 50, 'mu': 0.0, 'phi': 0.0}
            >>> controls = converter.convert(u, state)
            >>> print(controls['nx'], controls['nf'], controls['gamma'])
        """
        # Validate inputs
        if not all(k in current_state for k in ['V', 'mu', 'phi']):
            raise ValueError(
                f"current_state must contain [V, mu, phi], got {current_state.keys()}"
            )
        
        # Extract state variables
        mu = current_state['mu']
        phi = current_state['phi']
        
        # Precompute trigonometric functions
        cos_mu = np.cos(mu)
        sin_mu = np.sin(mu)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        # Build transformation matrix M (Equation 3, top part)
        M = np.array([
            [cos_mu * cos_phi, -sin_mu * cos_phi, -cos_mu * sin_phi],
            [cos_mu * sin_phi, -sin_mu * sin_phi,  cos_mu * cos_phi],
            [sin_mu,            cos_mu,             0.0             ]
        ])
        
        # Compute M⁻¹ · [ux/g, uy/g, uz·cos(μ)/g]ᵀ
        # Using numpy's efficient inverse
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            warnings.warn(
                f"Singular matrix M at μ={mu:.3f}, φ={phi:.3f}. "
                "Using pseudo-inverse."
            )
            M_inv = np.linalg.pinv(M)
        
        # Build input vector for transformation
        input_vec = np.array([
            u_xyz[0] / self.g,
            u_xyz[1] / self.g,
            u_xyz[2] * cos_mu / self.g
        ])
        
        # Compute Γ = M⁻¹ · input_vec + [sin(μ), cos(μ), 0]ᵀ (Equation 3)
        Gamma = M_inv @ input_vec + np.array([sin_mu, cos_mu, 0.0])
        
        # Extract control commands (Equations 4-6)
        nx = self._compute_nx(Gamma)
        gamma = self._compute_gamma(Gamma)
        nf = self._compute_nf(Gamma, gamma)
        
        # Apply limits (this is the source of truth)
        nx_clamped = np.clip(nx, self.nx_min, self.nx_max)
        nf_clamped = np.clip(nf, -self.nf_max, self.nf_max)
        gamma_clamped = np.clip(gamma, -self.gamma_max, self.gamma_max)
        
        return {
            'nx': float(nx_clamped),
            'nf': float(nf_clamped),
            'gamma': float(gamma_clamped)
        }
    
    def _compute_nx(self, Gamma: np.ndarray) -> float:
        """
        Compute longitudinal load factor (Equation 4).
        
        Args:
            Gamma: Equivalent vector [Γ₁, Γ₂, Γ₃]
        
        Returns:
            nx: Longitudinal load factor
        """
        return Gamma[0]
    
    def _compute_gamma(self, Gamma: np.ndarray) -> float:
        """
        Compute bank angle (Equation 5).
        
        Args:
            Gamma: Equivalent vector [Γ₁, Γ₂, Γ₃]
        
        Returns:
            gamma: Bank angle (rad)
        """
        return np.arctan2(Gamma[2], Gamma[1])
    
    def _compute_nf(self, Gamma: np.ndarray, gamma: float) -> float:
        """
        Compute normal load factor (Equation 6).
        
        Uses conditional logic to avoid division by zero:
        - If sin(γ) ≈ 0: use nf = Γ₂/cos(γ)
        - If cos(γ) ≈ 0: use nf = Γ₃/sin(γ)
        
        Args:
            Gamma: Equivalent vector [Γ₁, Γ₂, Γ₃]
            gamma: Bank angle (rad)
        
        Returns:
            nf: Normal load factor
        """
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)
        
        # Choose formula based on which denominator is larger (more stable)
        if abs(cos_gamma) > abs(sin_gamma):
            # cos(γ) is larger → use Equation 6 first form
            # Avoid division by zero
            cos_gamma_safe = max(abs(cos_gamma), self._gamma_eps) * np.sign(cos_gamma)
            nf = Gamma[1] / cos_gamma_safe
        else:
            # sin(γ) is larger → use Equation 6 second form
            # Avoid division by zero
            sin_gamma_safe = max(abs(sin_gamma), self._gamma_eps) * np.sign(sin_gamma)
            nf = Gamma[2] / sin_gamma_safe
        
        return nf
    
    def get_limits(self) -> Dict[str, Tuple[float, float]]:
        """
        Return control limits for reference.
        
        Returns:
            Dict with ranges: {
                'nx': (nx_min, nx_max),
                'nf': (-nf_max, nf_max),
                'gamma': (-gamma_max, gamma_max)
            }
        """
        return {
            'nx': (self.nx_min, self.nx_max),
            'nf': (-self.nf_max, self.nf_max),
            'gamma': (-self.gamma_max, self.gamma_max)
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ControlConverter(nx=[{self.nx_min}, {self.nx_max}], "
            f"nf=±{self.nf_max}, γ=±{self.gamma_max:.2f})"
        )
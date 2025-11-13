
from math import cos
import numpy as np
from math import sin


class UAV:
    """
    Modèle 6-DOF d'un UAV (Section II.B, équation 2)
    
    État: [x, y, z, V, μ, φ]
    Entrées: [nx, nf, γ]
    """
    
    def __init__(self, config, initial_state):
        # Paramètres physiques
        self.g = 9.8
        self.V_min = config['V_min']
        self.V_max = config['V_max']
        
        # État actuel (6 variables)
        self.x, self.y, self.z = initial_state['position']
        self.V = initial_state['V']
        self.mu = initial_state['mu']
        self.phi = initial_state['phi']
        

    # Elle prend l’état actuel du drone + les commandes du moment, 
    # et elle calcule le nouvel état après un petit pas de temps dt.
    
    def integrate_6dof(self, controls, dt):

        """
        Intègre équation (2) sur un pas de temps dt
        
        Args:
            controls: dict {nx, nf, gamma}
            dt: pas de temps (0.01s)
        """
        nx, nf, gamma = controls['nx'], controls['nf'], controls['gamma']
        
        # Équation (2) - Calcul des dérivées
        x_dot = self.V * cos(self.mu) * cos(self.phi)
        y_dot = self.V * cos(self.mu) * sin(self.phi)
        z_dot = self.V * sin(self.mu)
        
        V_dot = self.g * (nx - sin(self.mu))
        mu_dot = self.g * (nf*cos(gamma) - cos(self.mu)) / self.V
        phi_dot = self.g * nf * sin(gamma) / (self.V * cos(self.mu))
        
        # Intégration Euler
        self.x += x_dot * dt
        self.y += y_dot * dt
        self.z += z_dot * dt
        self.V += V_dot * dt
        self.mu += mu_dot * dt
        self.phi += phi_dot * dt
        
        # Contraintes physiques
        self._apply_constraints()
    
    def _apply_constraints(self):
        """Applique limites physiques"""
        self.V = np.clip(self.V, self.V_min, self.V_max)
        self.mu = np.clip(self.mu, -np.pi/2+0.01, np.pi/2-0.01)
        self.phi = (self.phi + np.pi) % (2*np.pi) - np.pi  # [-π, π]
    
    @property
    def position(self):
        """Retourne position cartésienne pour hawk/pigeon"""
        return np.array([self.x, self.y, self.z])
    
    @property
    def velocity(self):
        """Retourne vitesse cartésienne [vx, vy, vz]"""
        return np.array([
            self.V * cos(self.mu) * cos(self.phi),
            self.V * cos(self.mu) * sin(self.phi),
            self.V * sin(self.mu)
        ])
    
    def get_state(self):
        """État complet pour converter"""
        return {
            'V': self.V,
            'mu': self.mu,
            'phi': self.phi
        }
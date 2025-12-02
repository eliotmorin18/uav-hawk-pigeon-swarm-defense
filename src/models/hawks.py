import numpy as np
from uav_base import UAV  # Import de la classe mère


class Hawk(UAV):
    """
    Classe Hawk (Faucon) - UAV Défenseur.
    Hérite de UAV et implémente les mécanismes de chasse inspirés du faucon de Harris.
    """
    
    def __init__(self, x=0.0, y=0.0, z=0.0, V=100.0, mu=0.0, phi=0.0,
                 sensing_radius=1000.0, neighbor_radius=200.0):
        """
        Constructeur de la classe Hawk.
        
        Paramètres UAV (hérités):
        -------------------------
        x, y, z : float
            Position initiale (m)
        V : float
            Vitesse initiale (m/s)
        mu : float
            Angle de trajectoire initial (rad)
        phi : float
            Angle de cap initial (rad)
        
        Paramètres Hawk (spécifiques):
        ------------------------------
        sensing_radius : float
            Rayon de détection Rs pour filtrer les pigeons visibles (m)
            Utilisé pour définir Z1 dans les équations (7), (8), (9)
        neighbor_radius : float
            Rayon de voisinage Rnei pour les calculs de marge et densité (m)
            Utilisé pour définir Z2 et Z3 dans les équations (8) et (9)
        """
        # Appel du constructeur parent
        super().__init__(x, y, z, V, mu, phi)
        self.type = 'Hawk'
        self.index = -1 
        
        # Attributs de perception / sélection de cible
        self.sensing_radius = sensing_radius      # Rs dans l'article
        self.neighbor_radius = neighbor_radius    # Rnei dans l'article
        
        
        # Gestion de la cible
        self.current_target = None    # Référence au pigeon actuellement poursuivi

    
    def __repr__(self):
        """Représentation textuelle du Hawk."""
        target_info = f"target={self.current_target}" if self.target_locked else "no target"
        return (f"Hawk(pos=[{self.x:.1f}, {self.y:.1f}, {self.z:.1f}], "
                f"V={self.V:.1f}, {target_info})")
    
    def reset_target(self):
        """Réinitialise la cible actuelle."""
        self.current_target = None
        self.target_locked = False

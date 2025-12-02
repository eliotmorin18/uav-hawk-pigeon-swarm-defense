import numpy as np
from uav_base import UAV  # Import de la classe mère


class Hawk(UAV):
    """
    Classe Hawk (Faucon) - UAV Défenseur.
    Hérite de UAV et implémente les mécanismes de chasse inspirés du faucon de Harris.
    """
    
    def __init__(self, x=0.0, y=0.0, z=0.0, V=100.0, mu=0.0, phi=0.0,
                 sensing_radius=1000.0, neighbor_radius=100.0):
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


    def pigeon_in_sensing_radius (ar_pigeon,sensing_radius, hawk_coord):
        list_pigeon = []
        hawk_coord = np.array(hawk_coord)
        for pigeon in ar_pigeon:
            pigeon_pos = np.array(pigeon.state[:3])
            dif = hawk_coord - pigeon_pos
            distance = np.linalg.norm(dif)
            if distance <= sensing_radius:
                list_pigeon.append(pigeon)
        if len(list_pigeon) == 0:
            return None
    
        return np.array(list_pigeon)
    

    def closest_pigeon(ar_pigeon_in_sensing, hawk_coord):

        if ar_pigeon_in_sensing is None or len(ar_pigeon_in_sensing) == 0:
            return None
        
        hawk_coord = np.array (hawk_coord)
        position = np.array([pigeon.state[:3]for pigeon in ar_pigeon_in_sensing])

        dif= position - hawk_coord
        distance = np.linalg.norm(dif,axis=1)
        i_min = np.argmin(distance)

        return ar_pigeon_in_sensing[i_min]






    
    def potential_target_1(ar_pigeon,sensing_radius,hawk_coord):

        if ar_pigeon is None or len(ar_pigeon) == 0:
            return None
        
        pigeon = Hawk.pigeon_in_sensing_radius (ar_pigeon,sensing_radius, hawk_coord)
        target_1 = Hawk.closest_pigeon(pigeon,hawk_coord)

        return target_1


    




    def find_neighboor_4_1_pigeon(thispigeon, ar_pigeons_in_range,neighbor_radius):
        list_pigeon = []
        thispigeon = pigeon.state[:3]
        neighbors_candidates = [p for p in ar_pigeons_in_range if p is not thispigeon ]
        for index, pigeon in enumerate(neighbors_candidates):
            pigeon = pigeon.state[:3]
            pigeon = np.array(pigeon)
            dif = thispigeon - pigeon
            distance = np.linalg.norm(dif)
            if distance < neighbor_radius:
                list_pigeon.append(neighbors_candidates[index])
        if list_pigeon == []:
            return None
        return np.array(list_pigeon)
        


    def potential_target_2(ar_pigeon,sensing_radius,hawk_coord,neighbor_radius):
        pigeon_in_range = Hawk.pigeon_in_sensing_radius (ar_pigeon,sensing_radius, hawk_coord)
        if not pigeon_in_range or len(pigeon_in_range) == 0:
            return None
        
        periph = np.zeros(len(pigeon_in_range))
        hawk_coord = np.array(hawk_coord)
        for index, pigeon in enumerate(pigeon_in_range):
            neighbor_pigeons = Hawk.find_neighboor_4_1_pigeon(pigeon, pigeon_in_range, neighbor_radius)
            if neighbor_pigeons is None or len(neighbor_pigeons) == 0:
            # Pas de voisins -> périphéralité maximale (pigeon isolé)
                periph[index] = 1000.0

            else:
                N_m = neighbor_pigeons.shape[0]
                q_pigeon_sum = np.zeros(3)
            for neighbor in neighbor_pigeons:
                neighbor_pos = np.array(neighbor.state[:3])
                vector_hawk_to_neighbor = neighbor_pos - hawk_coord
                norm = np.linalg.norm(vector_hawk_to_neighbor)
                if norm > 0:
                    unit_vector = vector_hawk_to_neighbor / norm
                    q_pigeon_sum += unit_vector

            q_pigeon = q_pigeon_sum / N_m
            periph[index] = np.linalg.norm(q_pigeon)
        
        max_periph_index = np.argmax(periph)
        target_2 = pigeon_in_range[max_periph_index]

        return target_2


    def potential_target_3(ar_pigeon,sensing_radius,hawk_coord,neighbor_radius,Kpn):
        pigeon_in_range = Hawk.pigeon_in_sensing_radius (ar_pigeon,sensing_radius, hawk_coord)
        if not pigeon_in_range or len(pigeon_in_range) == 0:
            return None
        
        score_density = np.zeros(len(pigeon_in_range))
        for index, pigeon in enumerate(pigeon_in_range):
            neighbor_pigeons = Hawk.find_neighboor_4_1_pigeon(pigeon, pigeon_in_range, neighbor_radius)
            neighbor_pigeons = neighbor_pigeons[:,:3]
            N_m = neighbor_pigeons.shape[0]
            dist_2_center = np.sum(neighbor_pigeons)/ N_m
            dist_pig_neigh = np.norm(pigeon.state[:3]-dist_2_center)
        
            score_density[index] = dist_pig_neigh/Kpn
        
        max_index_score = np.argmax(score_density)
        return pigeon_in_range[max_index_score]
    

    def eval(target1, target2, target3, hawk_coord, hawk_velocity, D_C):
        L = [target1,target2,target3]
        scores = []
        for pigeon in L:
            if pigeon is None:
                scores.append(-np.inf)
                continue
            pigeon_coord = pigeon.state[:3]
            pigeon_velocity = pigeon.velocity_vector

            r_vec = pigeon_coord - hawk_coord
            r = np.linalg.norm(r_vec)

            if np.linalg.norm(hawk_velocity) > 0 and r > 0:
                cos_beta = np.dot(hawk_velocity, r_vec) / (np.linalg.norm(hawk_velocity) * r)
                beta = np.arccos(np.clip(cos_beta, -1, 1))
            else:
                beta = 0


            if np.linalg.norm(pigeon_velocity) > 0 and r > 0:
                cos_beta_p = np.dot(pigeon_velocity, r_vec) / (np.linalg.norm(pigeon_velocity) * r)
                beta_p = np.arccos(np.clip(cos_beta_p, -1, 1))
            else:
                beta_p = 0

            S_O = 1 - (beta + beta_p) / np.pi
            S_R = np.exp(-(r - D_C)**2 / (2 * D_C**2))
            S_OR = S_O * S_R
            scores.append(S_OR)

        best_index = np.argmax(scores)
        best_target = L[best_index]

        return best_target, scores[best_index]




    def __repr__(self):
        """Représentation textuelle du Hawk."""
        target_info = f"target={self.current_target}" if self.target_locked else "no target"
        return (f"Hawk(pos=[{self.x:.1f}, {self.y:.1f}, {self.z:.1f}], "
                f"V={self.V:.1f}, {target_info})")
    
    def reset_target(self):
        """Réinitialise la cible actuelle."""
        self.current_target = None
        self.target_locked = False

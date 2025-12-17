import numpy as np
from models.uav import UAV
import json
import os


class Hawk(UAV):
    """
    Hawk UAV implementing:
    - Target selection mechanisms (proximity, margin, density)
    - Harris' Hawk pursuit strategy (PN + PP)
    """

    def __init__(self, x, y, z, V=100.0, mu=0.0, phi=0.0,
                sensing_radius=150.0, neighbor_radius=100.0, capture_range=20.0):

        super().__init__(x, y, z, V, mu, phi)

        self.type = 'Hawk'
        self.sensing_radius = sensing_radius    #Rs
        self.neighbor_radius = neighbor_radius  #Rnei
        self.capture_range = capture_range      #DC

        self.t_go_blend = 1.0
        self.t_go_max = 3.0

        self.current_target = None
        self.target_locked = False


    @staticmethod
    def pigeon_in_sensing_radius(pigeons, sensing_radius, hawk_pos):
        """Return pigeons within the sensing radius Rs."""
        visibles = []
        hawk_pos = np.array(hawk_pos)

        for pigeon in pigeons:
            dif = hawk_pos - pigeon.state[:3]
            distance = np.linalg.norm(dif)
            if distance <= sensing_radius:
                visibles.append(pigeon)

        return visibles


    @staticmethod
    def closest_pigeon(pigeons, hawk_pos):
        """Proximity Criterion (Eq. 7)"""

        if len(pigeons) == 0:
            return None

        hawk_pos = np.array (hawk_pos)
        positions = np.array([pigeon.state[:3] for pigeon in pigeons])
        distances = np.linalg.norm(hawk_pos - positions, axis=1)
        i_min = np.argmin(distances)

        return pigeons[i_min]


    @staticmethod
    def neighbors_of(pigeon, pigeons, neighbor_radius):
        """Find neighbors of a pigeon within Rnei (except itself)."""

        neigh = []
        pigeon_pos = pigeon.state[:3]

        for p in pigeons :
            if p is pigeon :
                continue

            distance = np.linalg.norm(p.state[:3] - pigeon_pos) 
            if distance < neighbor_radius:
                neigh.append(p)

        return neigh


    @staticmethod
    def margin_target(pigeons, hawk_pos, sensing_radius, neighbor_radius):
        """
        Sélection selon le critère de marge (Margin Criterion).
        Équation (8) : T²_pigeon = arg max_j∈Z1 (angle marginal)

        L'angle marginal = angle entre :
        - Vecteur faucon→pigeon
        - Vecteur de périphéralité q^j_pigeon
        """

        pigeon_in_range = Hawk.pigeon_in_sensing_radius(pigeons, sensing_radius, hawk_pos)

        if not pigeon_in_range:
            return None

        marginal_angles = np.zeros(len(pigeon_in_range))
        hawk_pos = np.array(hawk_pos)

        for index, pigeon in enumerate(pigeon_in_range):
            neighbor_pigeons = Hawk.neighbors_of(pigeon, pigeon_in_range, neighbor_radius)

            if not neighbor_pigeons:
                marginal_angles[index] = np.pi
                continue


            N_m = len(neighbor_pigeons)
            q_pigeon_sum = np.zeros(3)

            for neighbor in neighbor_pigeons:
                neighbor_pos = neighbor.state[:3]
                vector_hawk_to_neighbor = neighbor_pos - hawk_pos
                norm = np.linalg.norm(vector_hawk_to_neighbor)

                if norm > 1e-9:
                    unit_vector = vector_hawk_to_neighbor / norm
                    q_pigeon_sum += unit_vector

            q_pigeon = q_pigeon_sum / N_m

            vector_hawk_to_pigeon = pigeon.state[:3] - hawk_pos
            norm_hawk_pigeon = np.linalg.norm(vector_hawk_to_pigeon)
            norm_q = np.linalg.norm(q_pigeon)

            if norm_hawk_pigeon < 1e-9 or norm_q < 1e-9:
                marginal_angles[index] = 0
                continue

            cos_angle = np.dot(vector_hawk_to_pigeon, q_pigeon) / (norm_hawk_pigeon * norm_q)
            marginal_angles[index] = np.arccos(np.clip(cos_angle, -1, 1))

        max_angle_index = np.argmax(marginal_angles)
        margin_target = pigeon_in_range[max_angle_index]

        return margin_target


    def density_target(pigeons, hawk_pos, sensing_radius, neighbor_radius):
        """
        Sélection selon le critère de densité (Density Criterion).
        Équation (9) : T³_pigeon = arg max_j∈Z1 (score_densité)

        score_densité = ||p^j_pigeon - p^j_c_pigeon|| / (1 + exp(-N_d))
        où p^j_c_pigeon = centre des voisins du pigeon j
        """
        pigeon_in_range = Hawk.pigeon_in_sensing_radius(pigeons, sensing_radius, hawk_pos)
        if not pigeon_in_range:
            return None

        density_scores = np.zeros(len(pigeon_in_range))

        for index, pigeon in enumerate(pigeon_in_range):
            neighbor_pigeons = Hawk.neighbors_of(pigeon, pigeon_in_range, neighbor_radius)

            if not neighbor_pigeons:
                density_scores[index] = 0.0
                continue

            N_d = len(neighbor_pigeons)  # Nombre de voisins

            # Calculer le centre de position des voisins : p^j_c_pigeon
            neighbor_positions = [n.state[:3] for n in neighbor_pigeons]
            center_position = np.mean(neighbor_positions, axis=0)

            # Distance du pigeon j au centre de ses voisins
            pigeon_pos = pigeon.state[:3]
            dist_center = np.linalg.norm(pigeon_pos - center_position)

            # Score de densité selon équation (9)
            density_scores[index] = dist_center / (1 + np.exp(-N_d))

            # Sélectionner le pigeon avec le score de densité MAXIMUM
        max_density_index = np.argmax(density_scores)
        density_target = pigeon_in_range[max_density_index]

        return density_target

    def compute_time_to_go(self, targeted_pigeon) :
        """
        Estimate time-to-go (t_go) until interception under
        constant-velocity assumption.
        """

        p_h = np.array(self.state[:3])
        p_p = np.array(targeted_pigeon.state[:3])

        v_h = np.array(self.velocity_vector)
        v_p = np.array(targeted_pigeon.velocity_vector)

        r_vec = p_p - p_h
        r = np.linalg.norm(r_vec)

        if r <= self.capture_range:
            return 0.0

        v_rel = v_p - v_h
        V_c = - np.dot(r, v_rel)

        if V_c <= 1e-3:
            return self.t_go_max

        t_go = (r - self.capture_range) / V_c

        return t_go
    
    def compute_intercept_time(self, pigeon):
        p_h = np.array(self.state[:3])
        p_p = np.array(pigeon.state[:3])
        v_p = np.array(pigeon.velocity_vector)
        V_h = np.linalg.norm(self.velocity_vector)

        r = p_p - p_h

        a = np.dot(v_p, v_p) - V_h**2
        b = 2 * np.dot(r, v_p)
        c = np.dot(r, r)

        disc = b*b - 4*a*c
        if disc < 0:
            return None

        t1 = (-b - np.sqrt(disc)) / (2*a)
        t2 = (-b + np.sqrt(disc)) / (2*a)

        t_candidates = [t for t in (t1, t2) if t > 0]

        return min(t_candidates) if t_candidates else None



    def predict_pigeon_position(self, targeted_pigeon, t_go):
        """
        Predict pigeon position after t_go seconds
        using constant-velocity model.
        """

        p_p = np.array(targeted_pigeon.state[:3])
        v_p = np.array(targeted_pigeon.velocity_vector)

        # Clamp time-to-go for safety
        t = np.clip(t_go, 0.0, self.t_go_max)

        p_pred = p_p + v_p * t

        return p_pred


    def get_effective_target_position(self, targeted_pigeon, p_pred, t_go):
        """
        Blend current and predicted position for smooth guidance.
        """

        p_now = np.array(targeted_pigeon.state[:3])

        # Blend factor grows with time-to-go
        alpha = np.clip(t_go / self.t_go_blend, 0.0, 1.0)

        p_target = (1 - alpha) * p_now + alpha * p_pred

        return p_target



    def choose_target(self, pigeons):
        """Return the chosen target among T1, T2, T3 using S_OR (Eq. 12)."""

        hawk_pos = np.array(self.state[:3])
        v_hawk = np.array(self.velocity_vector)
        DC = self.capture_range

        # Compute T1, T2, T3
        T1 = Hawk.closest_pigeon(
            Hawk.pigeon_in_sensing_radius(pigeons, self.sensing_radius, hawk_pos),
            hawk_pos
        )
        T2 = Hawk.margin_target(pigeons, hawk_pos, self.sensing_radius, self.neighbor_radius)
        T3 = Hawk.density_target(pigeons, hawk_pos, self.sensing_radius, self.neighbor_radius)

        candidates = [T1, T2, T3]

        # Score each candidate
        scores = []
        for pigeon in candidates:
            if pigeon is None:
                scores.append(-np.inf)
                continue

            pigeon_pos = np.array(pigeon.state[:3])
            v_pigeon = np.array(pigeon.velocity_vector)
            r = pigeon_pos - hawk_pos
            nr = np.linalg.norm(r)

            # Compute β and β_p
            def deviation_angle(a, b):
                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)
                if na < 1e-9 or nb < 1e-9:
                    return 0
                cosb = np.dot(a, b) / (na * nb)
                return np.arccos(np.clip(cosb, -1, 1))

            beta = deviation_angle(v_hawk, r)
            beta_p = deviation_angle(v_pigeon, r)

            S_O = 1 - (beta + beta_p) / np.pi
            S_R = np.exp(-(nr - DC)**2 / (2 * DC**2))
            S_OR = S_O * S_R
            scores.append(S_OR)

        best = candidates[np.argmax(scores)]
        return best


    def control(self, pigeon):
        """Return acceleration u according to PN + PP pursuit law."""

        p_hawk = np.array(self.state[:3])
        p_pigeon = np.array(pigeon.state[:3])

        v_hawk = np.array(self.velocity_vector)
        v_pigeon = np.array(pigeon.velocity_vector)

#        t_go = self.compute_time_to_go(pigeon)

#        p_pred = self.predict_pigeon_position(pigeon, t_go)

#        p_aim= self.get_effective_target_position(pigeon, p_pred, t_go)

        t_int = self.compute_intercept_time(pigeon)
        if t_int is not None:
            p_aim = p_pigeon + v_pigeon * t_int
        else:
            p_aim = p_pigeon


        

        r = p_aim - p_hawk
        nr = np.linalg.norm(r)

        omega = np.cross(r, (v_pigeon - v_hawk)) / (nr**2 + 1e-9)

        v_hawk_norm = np.linalg.norm(v_hawk)
        if v_hawk_norm < 1e-9 or nr < 1e-9:
            return np.zeros(3)

        cos_beta = np.dot(r, v_hawk) / ((nr * v_hawk_norm) + 1e-9)
        beta_scalar = np.arccos(np.clip(cos_beta, -1, 1))

        cross_r_v = np.cross(r, v_hawk)
        cross_norm = np.linalg.norm(cross_r_v)

        if cross_norm > 1e-9:
            beta_direction = cross_r_v / cross_norm
        else:
            beta_direction = np.zeros(3)

        beta_vec = beta_scalar * beta_direction
        beta_norm = beta_scalar

        K_PN = np.exp(-beta_norm)
        K_PP = 9.81 * beta_norm

        u_PN = K_PN * np.cross(omega, v_hawk)
        u_PP = K_PP * np.cross(beta_vec, v_hawk)

        u = u_PN - u_PP

        return u


    def __repr__(self):
        """Représentation textuelle du Hawk."""
        target_info = f"target={self.current_target}" if self.target_locked else "no target"
        return (f"Hawk(pos=[{self.x:.1f}, {self.y:.1f}, {self.z:.1f}], "
                f"V={self.V:.1f}, {target_info})")

    def reset_target(self):
        """Réinitialise la cible actuelle."""
        self.current_target = None
        self.target_locked = False

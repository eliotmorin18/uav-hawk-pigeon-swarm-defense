import json
import numpy as np
from models.hawks import Hawk
from models.pigeon import Pigeon
from core.game import Game


def load_parameters(config_path="config/parameters.json"):
    """Charge les paramètres depuis le fichier JSON."""
    with open(config_path, 'r') as f:
        return json.load(f)


def initialize_pigeons(params):
    """Crée les pigeons avec les paramètres."""
    pigeons = []
    pigeon_params = params["pigeon"]
    
    for pos in params["initial_positions"]["pigeons"]:
        x, y, z = pos
        target = np.array(params["target"])
        
        # Direction vers la cible
        direction = target - np.array([x, y, z])
        r = np.linalg.norm(direction)
        
        # mu: angle d'élévation (pitch)
        mu = np.arcsin(np.clip(direction[2] / r, -1, 1))
        
        # phi: angle de cap (yaw)
        phi = np.arctan2(direction[1], direction[0])
        
        print(f"Initial mu={np.degrees(mu):.1f}°, phi={np.degrees(phi):.1f}°")
        
        pigeon = Pigeon(
            x=x, y=y, z=z,
            V=pigeon_params["max_speed"],
            mu=mu, phi=phi,
            params={
                "k1": pigeon_params["k1"],
                "Re": pigeon_params["Re"],
                "Ra": pigeon_params["Ra"]
            }
        )
        pigeons.append(pigeon) 

    return pigeons


def initialize_hawks(params):
    """Crée les hawks avec les paramètres."""
    hawks = []
    hawk_params = params["hawk"]
    pigeons_pos = params["initial_positions"]["pigeons"]
    
    for hawk_pos_init in params["initial_positions"]["hawks"]:
        x, y, z = hawk_pos_init
        
        # Trouver le pigeon le plus proche
        closest_dist = float('inf')
        closest_pigeon_pos = None
        for pigeon_pos in pigeons_pos:
            dist = np.linalg.norm(np.array(hawk_pos_init) - np.array(pigeon_pos))
            if dist < closest_dist:
                closest_dist = dist
                closest_pigeon_pos = pigeon_pos
        
        # Direction vers le pigeon le plus proche
        direction = np.array(closest_pigeon_pos) - np.array(hawk_pos_init)
        r = np.linalg.norm(direction)
        
        mu = np.arcsin(np.clip(direction[2] / r, -1, 1))
        phi = np.arctan2(direction[1], direction[0])

        print(f"Hawk initial mu={np.degrees(mu):.1f}°, phi={np.degrees(phi):.1f}°")
        print(f"Hawk initial velocity_vector = {np.array([400*np.cos(mu)*np.cos(phi), 400*np.cos(mu)*np.sin(phi), 400*np.sin(mu)])}")
        
        print(f"Hawk mu={np.degrees(mu):.1f}°, phi={np.degrees(phi):.1f}°")
        
        hawk = Hawk(
            x=x, y=y, z=z,
            V=hawk_params["max_speed"],
            mu=mu, phi=phi,
            sensing_radius=hawk_params["Rs"],
            neighbor_radius=hawk_params["Rnei"],
            capture_range=20.0
        )
        hawks.append(hawk)
    
    return hawks





def main():
    """Fonction principale pour lancer la simulation."""
    
    # Charger les paramètres
    params = load_parameters("config/parameters.json")
    sim_params = params["simulation"]
    rules_params = params["rules"]
    
    # Initialiser les pigeons et hawks
    pigeons = initialize_pigeons(params)
    hawks = initialize_hawks(params)
    target = np.array(params["target"])
    
    print("=" * 60)
    print("SIMULATION HAWKS vs PIGEONS")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Pigeons: {len(pigeons)}")
    print(f"  Hawks: {len(hawks)}")
    print(f"  Target: {target}")
    print(f"  Duration: {sim_params['T']}s")
    print(f"  Time step: {sim_params['dt']}s")
    print()
    
    for i, p in enumerate(pigeons):
        print(f"  Pigeon {i}: {p}")
    for i, h in enumerate(hawks):
        print(f"  Hawk {i}: {h}")
    
    # Créer le jeu
    game = Game(
        hawks=hawks,
        pigeons=pigeons,
        target=target,
        dt=sim_params["dt"],
        capture_radius= rules_params["capture_radius"]
    )
    
    # Lancer la simulation
    print("\n" + "=" * 60)
    print("Lancement de la simulation...")
    print("=" * 60 + "\n")
    
    try:
        result = game.run(sim_params["T"])
        
        print("\n" + "=" * 60)
        print(f"Simulation terminée: {result}")
        print("=" * 60)
        print(f"\nRésultats finaux:")
        print(f"  Pigeons restants: {sum(game.pigeon_alive)}/{len(pigeons)}")
        print(f"  Temps écoulé: {game.time:.2f}s")
        
        # Afficher les positions finales
        print(f"\nPositions finales:")
        for i, p in enumerate(pigeons):
            if game.pigeon_alive[i]:
                print(f"  Pigeon {i}: {p.state[:3]}")
        for i, h in enumerate(hawks):
            print(f"  Hawk {i}: {h.state[:3]}")
        
        # Sauvegarder les trajectoires et positions pour affichage
        trajectoires_data = {
            "time": [],
            "hawks": [[] for _ in hawks],
            "pigeons": [[] for _ in pigeons],
            "metadata": {
                "num_hawks": len(hawks),
                "num_pigeons": len(pigeons),
                "target": target.tolist(),
                "simulation_time": game.time,
                "total_steps": len(game.trajectories["hawks"][0]) if hawks else 0
            }
        }
        
        # Récupérer les trajectoires stockées dans game
        for i, hawk_traj in enumerate(game.trajectories["hawks"]):
            trajectoires_data["hawks"][i] = [pos.tolist() for pos in hawk_traj]
        
        for i, pigeon_traj in enumerate(game.trajectories["pigeons"]):
            trajectoires_data["pigeons"][i] = [pos.tolist() for pos in pigeon_traj]
        
        # Générer les timestamps
        num_steps = len(game.trajectories["hawks"][0]) if hawks else len(game.trajectories["pigeons"][0])
        trajectoires_data["time"] = [t * sim_params["dt"] for t in range(num_steps)]
        
        # Sauvegarder en JSON
        with open("trajectoire.json", "w") as f:
            json.dump(trajectoires_data, f, indent=2)
        
        print(f"\n✓ Trajectoires sauvegardées dans 'trajectoire.json'")
        print(f"  Total de {num_steps} steps")
        print(f"  Hawks sauvegardés: {len(trajectoires_data['hawks'])}")
        print(f"  Pigeons sauvegardés: {len(trajectoires_data['pigeons'])}")
    
    except Exception as e:
        print(f"\n✗ ERREUR lors de la simulation:")
        print(f"  {type(e).__name__}: {e}")
        print("\nTraceback détaillé:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
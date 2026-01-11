import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def load_trajectories(filepath="trajectoire.json"):
    """Charge les trajectoires depuis le fichier JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_3d_trajectories(data):
    """Affiche les trajectoires en 3D."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Couleurs pour les hawks et pigeons
    hawk_colors = plt.cm.Reds(np.linspace(0.4, 1, len(data["hawks"])))
    pigeon_colors = plt.cm.Blues(np.linspace(0.4, 1, len(data["pigeons"])))
    
    # Afficher les trajectoires des hawks
    for i, hawk_traj in enumerate(data["hawks"]):
        if hawk_traj:
            traj = np.array(hawk_traj)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                   color=hawk_colors[i], linewidth=2, label=f"Hawk {i}")
            # Point initial
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                      color=hawk_colors[i], s=100, marker='o', edgecolors='black')
            # Point final
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                      color=hawk_colors[i], s=100, marker='X', edgecolors='black')
    
    # Afficher les trajectoires des pigeons
    for i, pigeon_traj in enumerate(data["pigeons"]):
        if pigeon_traj:
            traj = np.array(pigeon_traj)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                   color=pigeon_colors[i], linewidth=2, label=f"Pigeon {i}")
            # Point initial
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                      color=pigeon_colors[i], s=100, marker='o', edgecolors='black')
            # Point final
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                      color=pigeon_colors[i], s=100, marker='X', edgecolors='black')
    
    # Afficher la cible
    target = np.array(data["metadata"]["target"])
    ax.scatter(target[0], target[1], target[2], 
              color='green', s=200, marker='*', edgecolors='black', 
              label='Target', zorder=5)
    
    # Labels et titre
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Trajectoires Hawks vs Pigeons')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_2d_projections(data):
    """Affiche les projections 2D (XY, XZ, YZ)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    hawk_colors = plt.cm.Reds(np.linspace(0.4, 1, len(data["hawks"])))
    pigeon_colors = plt.cm.Blues(np.linspace(0.4, 1, len(data["pigeons"])))
    target = np.array(data["metadata"]["target"])
    
    projections = [
        (axes[0], 0, 1, "X (m)", "Y (m)", "XY Projection"),
        (axes[1], 0, 2, "X (m)", "Z (m)", "XZ Projection"),
        (axes[2], 1, 2, "Y (m)", "Z (m)", "YZ Projection")
    ]
    
    for ax, x_idx, y_idx, xlabel, ylabel, title in projections:
        # Hawks
        for i, hawk_traj in enumerate(data["hawks"]):
            if hawk_traj:
                traj = np.array(hawk_traj)
                ax.plot(traj[:, x_idx], traj[:, y_idx], 
                       color=hawk_colors[i], linewidth=1.5, alpha=0.7)
                ax.scatter(traj[0, x_idx], traj[0, y_idx], 
                          color=hawk_colors[i], s=50, marker='o')
                ax.scatter(traj[-1, x_idx], traj[-1, y_idx], 
                          color=hawk_colors[i], s=50, marker='X')
        
        # Pigeons
        for i, pigeon_traj in enumerate(data["pigeons"]):
            if pigeon_traj:
                traj = np.array(pigeon_traj)
                ax.plot(traj[:, x_idx], traj[:, y_idx], 
                       color=pigeon_colors[i], linewidth=1.5, alpha=0.7)
                ax.scatter(traj[0, x_idx], traj[0, y_idx], 
                          color=pigeon_colors[i], s=50, marker='o')
                ax.scatter(traj[-1, x_idx], traj[-1, y_idx], 
                          color=pigeon_colors[i], s=50, marker='X')
        
        # Target
        ax.scatter(target[x_idx], target[y_idx], 
                  color='green', s=150, marker='*', edgecolors='black', zorder=5)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_statistics(data):
    """Affiche les statistiques de la simulation."""
    metadata = data["metadata"]
    
    print("=" * 60)
    print("STATISTIQUES DE SIMULATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Hawks: {metadata['num_hawks']}")
    print(f"  Pigeons: {metadata['num_pigeons']}")
    print(f"  Target: {metadata['target']}")
    print(f"  Simulation time: {metadata['simulation_time']:.2f}s")
    print(f"  Total steps: {metadata['total_steps']}")
    
    # Trajectoires non-vides
    hawks_alive = sum(1 for h in data["hawks"] if h)
    pigeons_alive = sum(1 for p in data["pigeons"] if p)
    
    print(f"\nRésultats:")
    print(f"  Hawks restants: {hawks_alive}/{metadata['num_hawks']}")
    print(f"  Pigeons restants: {pigeons_alive}/{metadata['num_pigeons']}")
    
    # Distance parcourue
    print(f"\nDistances parcourues:")
    for i, hawk_traj in enumerate(data["hawks"]):
        if hawk_traj:
            traj = np.array(hawk_traj)
            dist = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
            print(f"  Hawk {i}: {dist:.2f} m")
    
    for i, pigeon_traj in enumerate(data["pigeons"]):
        if pigeon_traj:
            traj = np.array(pigeon_traj)
            dist = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
            print(f"  Pigeon {i}: {dist:.2f} m")
    
    print()


def main():
    """Fonction principale."""
    try:
        # Charger les trajectoires
        data = load_trajectories("trajectoire.json")
        
        # Afficher les statistiques
        print_statistics(data)
        
        # Créer et afficher les graphiques
        print("Génération des graphiques...")
        
        fig1 = plot_3d_trajectories(data)
        fig2 = plot_2d_projections(data)
        
        # Sauvegarder les figures
        fig1.savefig("trajectoires_3d.png", dpi=150, bbox_inches='tight')
        fig2.savefig("trajectoires_2d.png", dpi=150, bbox_inches='tight')
        
        print("✓ Graphiques sauvegardés:")
        print("  - trajectoires_3d.png")
        print("  - trajectoires_2d.png")
        
        # Afficher les graphiques
        plt.show()
        
    except FileNotFoundError:
        print("✗ Erreur: fichier 'trajectoire.json' non trouvé")
        print("  Assurez-vous d'avoir lancé la simulation d'abord")
    except Exception as e:
        print(f"✗ Erreur: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
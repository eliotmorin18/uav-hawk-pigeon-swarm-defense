import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D


def load_trajectories(filepath="trajectoire.json"):
    """Charge les trajectoires depuis le fichier JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_video(trajectories_file="trajectoire.json", output_file="simulation.mp4", fps=30):
    """Crée une vidéo 3D des trajectoires."""
    
    # Charger les données
    data = load_trajectories(trajectories_file)
    
    hawks = data["hawks"]
    pigeons = data["pigeons"]
    target = np.array(data["metadata"]["target"])
    times = data["time"]
    
    num_steps = len(times)
    
    print(f"Création de la vidéo...")
    print(f"  Steps: {num_steps}")
    print(f"  Hawks: {len(hawks)}")
    print(f"  Pigeons: {len(pigeons)}")
    print(f"  Durée: {times[-1]:.2f}s")
    
    # Créer la figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculer les limites
    all_positions = []
    for hawk_traj in hawks:
        all_positions.extend(hawk_traj)
    for pigeon_traj in pigeons:
        all_positions.extend(pigeon_traj)
    all_positions.append(target.tolist())
    
    all_positions = np.array(all_positions)
    margin = 50
    
    x_min, x_max = all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin
    y_min, y_max = all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin
    z_min, z_max = all_positions[:, 2].min() - margin, all_positions[:, 2].max() + margin
    
    # Initialiser les plots
    hawk_scatters = []
    pigeon_scatters = []
    hawk_trails = []
    pigeon_trails = []
    
    for i in range(len(hawks)):
        scatter = ax.scatter([], [], [], c='red', marker='^', s=100, label=f'Hawk {i}' if i == 0 else '')
        hawk_scatters.append(scatter)
        trail, = ax.plot([], [], [], 'r-', alpha=0.3, linewidth=1)
        hawk_trails.append(trail)
    
    for i in range(len(pigeons)):
        scatter = ax.scatter([], [], [], c='blue', marker='o', s=80, label=f'Pigeon {i}' if i == 0 else '')
        pigeon_scatters.append(scatter)
        trail, = ax.plot([], [], [], 'b-', alpha=0.3, linewidth=1)
        pigeon_trails.append(trail)
    
    # Cible
    ax.scatter(*target, c='green', marker='*', s=300, label='Target', edgecolors='black', linewidths=2)
    
    # Configuration des axes
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(loc='upper right')
    
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    
    def init():
        """Initialisation de l'animation."""
        for scatter in hawk_scatters + pigeon_scatters:
            scatter._offsets3d = ([], [], [])
        for trail in hawk_trails + pigeon_trails:
            trail.set_data([], [])
            trail.set_3d_properties([])
        time_text.set_text('')
        return hawk_scatters + pigeon_scatters + hawk_trails + pigeon_trails + [time_text]
    
    def update(frame):
        """Met à jour l'animation pour chaque frame."""
        # Mettre à jour les positions actuelles
        for i, scatter in enumerate(hawk_scatters):
            if frame < len(hawks[i]):
                pos = np.array(hawks[i][frame])
                scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
        
        for i, scatter in enumerate(pigeon_scatters):
            if frame < len(pigeons[i]):
                pos = np.array(pigeons[i][frame])
                scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
        
        # Afficher TOUTE la trajectoire depuis le début
        for i, trail in enumerate(hawk_trails):
            if frame < len(hawks[i]):
                # Trajectoire complète du début jusqu'à la frame actuelle
                traj = np.array(hawks[i][0:frame+1])
                if len(traj) > 0:
                    trail.set_data(traj[:, 0], traj[:, 1])
                    trail.set_3d_properties(traj[:, 2])
        
        for i, trail in enumerate(pigeon_trails):
            if frame < len(pigeons[i]):
                # Trajectoire complète du début jusqu'à la frame actuelle
                traj = np.array(pigeons[i][0:frame+1])
                if len(traj) > 0:
                    trail.set_data(traj[:, 0], traj[:, 1])
                    trail.set_3d_properties(traj[:, 2])
        
        # Mettre à jour le temps
        time_text.set_text(f'Temps: {times[frame]:.2f}s')
        
        return hawk_scatters + pigeon_scatters + hawk_trails + pigeon_trails + [time_text]
    
    # Créer l'animation
    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=num_steps, interval=1000/fps,
        blit=False, repeat=True
    )
    
    # Sauvegarder la vidéo
    print(f"\nSauvegarde de la vidéo '{output_file}'...")
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(output_file, writer=writer)
    
    print(f"✓ Vidéo sauvegardée avec succès!")
    print(f"  Fichier: {output_file}")
    print(f"  FPS: {fps}")
    print(f"  Durée: {num_steps/fps:.2f}s")
    
    plt.close()


if __name__ == "__main__":
    # Créer la vidéo
    create_video(
        trajectories_file="trajectoire.json",
        output_file="simulation.mp4",
        fps=30
    )
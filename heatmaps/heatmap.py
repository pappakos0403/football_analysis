import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2D focipálya rajzolása
def draw_football_pitch(ax):
    # Pálya méretei [m]
    pitch_length = 105
    pitch_width = 68

    # Zöld háttér
    ax.set_facecolor("green")

    # Oldalvonal és alapvonal
    ax.plot([0, 0, pitch_length, pitch_length, 0], [0, pitch_width, pitch_width, 0, 0], color="white", linewidth=2)

    # Félpálya vonala
    ax.axvline(pitch_length / 2, color="white", linewidth=2)

    # Kezdőkör
    centre_circle = plt.Circle((pitch_length / 2, pitch_width / 2), 9.15, color="white", fill=False, linewidth=2)
    ax.add_patch(centre_circle)

    # Büntetőterületek
    for x in [0, pitch_length]:
        penalty_box_length = 16.5
        penalty_box_width = 40.32
        if x == 0:
            ax.plot([x + penalty_box_length]*2, [(pitch_width - penalty_box_width)/2, (pitch_width + penalty_box_width)/2], color="white", linewidth=2)
            ax.plot([x, x + penalty_box_length], [(pitch_width - penalty_box_width)/2]*2, color="white", linewidth=2)
            ax.plot([x, x + penalty_box_length], [(pitch_width + penalty_box_width)/2]*2, color="white", linewidth=2)
        else:
            ax.plot([x - penalty_box_length]*2, [(pitch_width - penalty_box_width)/2, (pitch_width + penalty_box_width)/2], color="white", linewidth=2)
            ax.plot([x, x - penalty_box_length], [(pitch_width - penalty_box_width)/2]*2, color="white", linewidth=2)
            ax.plot([x, x - penalty_box_length], [(pitch_width + penalty_box_width)/2]*2, color="white", linewidth=2)

    # Kapuk
    goal_box_length = 5.5
    goal_box_width = 18.32
    for x in [0, pitch_length]:
        if x == 0:
            ax.plot([x + goal_box_length]*2, [(pitch_width - goal_box_width)/2, (pitch_width + goal_box_width)/2], color="white", linewidth=2)
            ax.plot([x, x + goal_box_length], [(pitch_width - goal_box_width)/2]*2, color="white", linewidth=2)
            ax.plot([x, x + goal_box_length], [(pitch_width + goal_box_width)/2]*2, color="white", linewidth=2)
        else:
            ax.plot([x - goal_box_length]*2, [(pitch_width - goal_box_width)/2, (pitch_width + goal_box_width)/2], color="white", linewidth=2)
            ax.plot([x, x - goal_box_length], [(pitch_width - goal_box_width)/2]*2, color="white", linewidth=2)
            ax.plot([x, x - goal_box_length], [(pitch_width + goal_box_width)/2]*2, color="white", linewidth=2)

    # Középpont
    centre_spot = plt.Circle((pitch_length / 2, pitch_width / 2), 0.2, color="white")
    ax.add_patch(centre_spot)

    # Méretarány beállítása
    ax.set_xlim(0, pitch_length)
    ax.set_ylim(0, pitch_width)
    ax.set_aspect('equal')


# Hőtérkép generálása játékosokra
def generate_player_heatmaps(player_coordinates: list[dict], output_dir="heatmaps//heatmap_images", min_ratio=0.5):
    os.makedirs(output_dir, exist_ok=True)

    # Játékos pozícióinak tárolása
    player_presence = {}
    player_positions = {}

    total_frames = len(player_coordinates)

    # Játékos ID-k gyűjtése és pozíciók tárolása
    for frame_data in player_coordinates:
        for player_id, coord in frame_data.items():
            if player_id not in player_presence:
                player_presence[player_id] = 0
                player_positions[player_id] = []
            player_presence[player_id] += 1
            player_positions[player_id].append(coord)

    # Csak azok a játékosok, akik legalább a frame-ek 50%-ában jelen vannak
    for player_id, count in player_presence.items():
        if count >= total_frames * min_ratio:
            coords = np.array(player_positions[player_id])
            x, y = coords[:, 0], coords[:, 1]

            # Hőtérkép generálása
            fig, ax = plt.subplots(figsize=(12, 8))
            draw_football_pitch(ax)  # Pálya háttér
            sns.kdeplot(x=x, y=y, fill=True, cmap="hot", bw_adjust=0.5, thresh=0.05, alpha=0.7)
            ax.set_title(f"Hőtérkép - Játékos {player_id}", fontsize=14)
            ax.set_xlabel("Pálya hossza (m)")
            ax.set_ylabel("Pálya szélessége (m)")

            # Mentés fájlba
            output_path = os.path.join(output_dir, f"heatmap_{player_id}.png")
            plt.savefig(output_path, dpi=150)
            plt.close()
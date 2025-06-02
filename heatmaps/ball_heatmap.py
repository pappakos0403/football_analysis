import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Focipálya kirajzolása
def draw_football_pitch(ax):
    pitch_length = 105
    pitch_width = 68

    ax.set_facecolor("green")
    ax.plot([0, 0, pitch_length, pitch_length, 0], [0, pitch_width, pitch_width, 0, 0], color="white", linewidth=2)
    ax.axvline(pitch_length / 2, color="white", linewidth=2)

    # Kezdőkör
    centre_circle = plt.Circle((pitch_length / 2, pitch_width / 2), 9.15, color="white", fill=False, linewidth=2)
    ax.add_patch(centre_circle)

    # Kapuk és tizenhatos
    for x in [0, pitch_length]:
        penalty_box_length = 16.5
        penalty_box_width = 40.32
        goal_box_length = 5.5
        goal_box_width = 18.32

        # Büntetőterület
        if x == 0:
            ax.plot([x + penalty_box_length] * 2, [(pitch_width - penalty_box_width)/2, (pitch_width + penalty_box_width)/2], color="white", linewidth=2)
            ax.plot([x, x + penalty_box_length], [(pitch_width - penalty_box_width)/2] * 2, color="white", linewidth=2)
            ax.plot([x, x + penalty_box_length], [(pitch_width + penalty_box_width)/2] * 2, color="white", linewidth=2)
        else:
            ax.plot([x - penalty_box_length] * 2, [(pitch_width - penalty_box_width)/2, (pitch_width + penalty_box_width)/2], color="white", linewidth=2)
            ax.plot([x, x - penalty_box_length], [(pitch_width - penalty_box_width)/2] * 2, color="white", linewidth=2)
            ax.plot([x, x - penalty_box_length], [(pitch_width + penalty_box_width)/2] * 2, color="white", linewidth=2)

        # Kapuk
        if x == 0:
            ax.plot([x + goal_box_length]*2, [(pitch_width - goal_box_width)/2, (pitch_width + goal_box_width)/2], color="white", linewidth=2)
            ax.plot([x, x + goal_box_length], [(pitch_width - goal_box_width)/2]*2, color="white", linewidth=2)
            ax.plot([x, x + goal_box_length], [(pitch_width + goal_box_width)/2]*2, color="white", linewidth=2)
        else:
            ax.plot([x - goal_box_length]*2, [(pitch_width - goal_box_width)/2, (pitch_width + goal_box_width)/2], color="white", linewidth=2)
            ax.plot([x, x - goal_box_length], [(pitch_width - goal_box_width)/2]*2, color="white", linewidth=2)
            ax.plot([x, x - goal_box_length], [(pitch_width + goal_box_width)/2]*2, color="white", linewidth=2)

    centre_spot = plt.Circle((pitch_length / 2, pitch_width / 2), 0.2, color="white")
    ax.add_patch(centre_spot)

    ax.set_xlim(0, pitch_length)
    ax.set_ylim(0, pitch_width)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def generate_ball_heatmap(ball_coordinates: list[tuple[float, float]], output_path="heatmaps/heatmap_images/ball_heatmap.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Koordináták kiszűrése, csak ahol nem None
    filtered_coords = [coord for coord in ball_coordinates if coord is not None]

    if not filtered_coords:
        print("Nincs érvényes labdakoordináta a hőtérkép generálásához!")
        return

    coords = np.array(filtered_coords)
    x = coords[:, 0]
    y = coords[:, 1]

    # Ábra létrehozása
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_football_pitch(ax)
    sns.kdeplot(x=x, y=y, fill=True, cmap="hot", bw_adjust=0.5, thresh=0.05, alpha=0.7)
    ax.set_title("Labda hőtérképe", fontsize=14)
    ax.set_xlabel("Pálya hossza (m)")
    ax.set_ylabel("Pálya szélessége (m)")

    # Mentés fájlba
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Labda hőtérkép mentve: {output_path}")
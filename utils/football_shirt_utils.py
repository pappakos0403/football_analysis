import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

# Mezszín meghatározása
def get_valid_player_colors(valid_players, track_id_to_team, team1_color, team2_color):

    player_colors = {}

    for track_id in valid_players:
        if track_id in track_id_to_team:
            team = track_id_to_team[track_id]
            if team == 1:
                player_colors[track_id] = rgb_to_normalized(team1_color)
            elif team == 2:
                player_colors[track_id] = rgb_to_normalized(team2_color)
        else:
            # Ha nincs csapat hozzárendelve → szürke szín normalizálva
            player_colors[track_id] = rgb_to_normalized((128, 128, 128))

    return player_colors

# Mezképek lementése a megfelelő mappába
def save_all_jersey_images(valid_player_colors, track_id_to_team, video_name: str):

    base_dir = f"output_videos/annotated_{video_name}/statistics"
    team1_dir = os.path.join(base_dir, "team1shirt_images")
    team2_dir = os.path.join(base_dir, "team2shirt_images")
    os.makedirs(team1_dir, exist_ok=True)
    os.makedirs(team2_dir, exist_ok=True)

    for track_id, color in valid_player_colors.items():
        team = track_id_to_team.get(track_id)
        if team == 1:
            save_path = os.path.join(team1_dir, f"{track_id}.png")
        elif team == 2:
            save_path = os.path.join(team2_dir, f"{track_id}.png")
        else:
            continue

        # Mezkép mentése
        fig, ax = plt.subplots(figsize=(3, 4))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 14)
        ax.axis('off')

        ax.add_patch(patches.Rectangle((3, 5), 4, 6, linewidth=1, edgecolor='black', facecolor=color))  # törzs
        ax.add_patch(patches.Rectangle((1.5, 8), 1.5, 3, linewidth=1, edgecolor='black', facecolor=color))  # bal ujj
        ax.add_patch(patches.Rectangle((7, 8), 1.5, 3, linewidth=1, edgecolor='black', facecolor=color))  # jobb ujj
        ax.add_patch(patches.Rectangle((4.25, 11), 1.5, 0.7, linewidth=1, edgecolor='black', facecolor='black'))  # nyak

        # Mezszám ráírása kontúrral a mellkasi részre
        number_str = str(track_id)
        text = ax.text(5, 8.5, number_str,
                       fontsize=20,
                       ha='center',
                       va='center',
                       color='white')

        # Fekete kontúr a mezszám köré
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()
        ])

        plt.tight_layout()
        plt.savefig(save_path, transparent=True)
        plt.close()

def rgb_to_normalized(rgb):
    return tuple([c / 255.0 for c in rgb])
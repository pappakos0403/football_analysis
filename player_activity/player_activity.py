import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Játékos aktivitási statisztikák generálása
def generate_player_activity_summary(player_coordinates_list, speed_estimator, tracker=None, output_dir="statistics"):
    os.makedirs(output_dir, exist_ok=True)

    # Játékosok jelenlétének számlálása
    player_presence = {}
    total_frames = len(player_coordinates_list)
    for frame_data in player_coordinates_list:
        for track_id in frame_data:
            player_presence[track_id] = player_presence.get(track_id, 0) + 1

    # Csak azok a játékosok, akik a frame-ek legalább 50%-ában jelen voltak
    valid_players = [track_id for track_id, count in player_presence.items() if count >= total_frames * 0.5]

    # Játékos statisztikák külön csapatonként
    player_stats_team1 = {}
    player_stats_team2 = {}
    for track_id in valid_players:
        player_info = speed_estimator.player_data.get(track_id, {})
        smoothed_list = player_info.get("smoothed_list", [])
        if not smoothed_list:
            continue
        smoothed_kmh = [s * 3.6 for s in smoothed_list]  # m/s → km/h
        avg_speed_kmh = np.mean(smoothed_kmh)
        max_speed_kmh = max(smoothed_kmh)
        total_distance_m = speed_estimator.get_player_distance_m(track_id)
        team_id = player_info.get("team_id", None)
        if team_id == 1:
            player_stats_team1[track_id] = (avg_speed_kmh, max_speed_kmh, total_distance_m)
        elif team_id == 2:
            player_stats_team2[track_id] = (avg_speed_kmh, max_speed_kmh, total_distance_m)

    # Csapatszínek lekérése a tracker példányból
    if tracker and hasattr(tracker, 'team1_color') and hasattr(tracker, 'team2_color'):
        team1_color_rgb = tuple(np.array(tracker.team1_color) / 255.0)
        team2_color_rgb = tuple(np.array(tracker.team2_color) / 255.0)
    else:
        team1_color_rgb = (0.0, 0.0, 1.0)  # alapértelmezett kék
        team2_color_rgb = (1.0, 0.5, 0.0)  # alapértelmezett narancs

    # Egy adott csapat játékosainak statisztikáit megjelenítő grafikon
    def plot_team_stats(player_stats, team_label):
        if not player_stats:
            return

        track_ids = list(player_stats.keys())
        avg_speeds = [player_stats[pid][0] for pid in track_ids]
        max_speeds = [player_stats[pid][1] for pid in track_ids]
        distances = [player_stats[pid][2] for pid in track_ids]

        x = np.arange(len(track_ids))
        fig, ax = plt.subplots(figsize=(12, 6))

        # Átlagos sebesség (kék)
        ax.scatter(x, avg_speeds, label='Átlagos sebesség', color='blue')
        for i, val in enumerate(avg_speeds):
            ax.text(x[i], val + 2.5, f"{val:.1f} km/h", ha='center', color='blue')

        # Max sebesség (narancssárga)
        ax.scatter(x, max_speeds, label='Max. sebesség', color='orange')
        for i, val in enumerate(max_speeds):
            ax.text(x[i], val + 2.5, f"{val:.1f} km/h", ha='center', color='orange')

        # Megtett távolság (zöld)
        ax.scatter(x, distances, label='Megtett távolság', color='green')
        for i, val in enumerate(distances):
            ax.text(x[i], val + 2.5, f"{val:.1f} m", ha='center', color='green')

        ax.set_xlabel('Játékos ID')
        ax.set_ylabel('Érték')

        # Cím + négyzet hozzáadása
        title_text = f"{team_label} játékosainak aktivitási statisztikái"
        ax.set_title(title_text, pad=40)
        team_color_for_patch = team1_color_rgb if team_label == "Team1" else team2_color_rgb

        # --- Csapatszínű négyzet beállításai ---
        square_side_points = 18.0  # négyzet mérete pontban
        fig_width_points = fig.get_figwidth() * fig.dpi
        fig_height_points = fig.get_figheight() * fig.dpi
        rect_width_fig_coords = square_side_points / fig_width_points
        rect_height_fig_coords = square_side_points / fig_height_points

        # Vízszintes pozíció (a cím bal széléhez arányosan)
        square_x_bottom_left = 0.5 - (rect_width_fig_coords / 2)
        square_x_bottom_left += 0.021

        # Függőleges pozíció (a cím szintjéhez igazítva)
        target_center_y_fig_coords = 0.915
        square_y_bottom_left = target_center_y_fig_coords - (rect_height_fig_coords / 2.0)

        rect = Rectangle((square_x_bottom_left, square_y_bottom_left), 
                         rect_width_fig_coords, rect_height_fig_coords,
                         facecolor=team_color_for_patch, edgecolor='black', linewidth=1,
                         transform=fig.transFigure, clip_on=False)
        fig.patches.append(rect)

        ax.set_xticks(x)
        ax.set_xticklabels([str(pid) for pid in track_ids])
        ax.grid(True)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"activity_summary_{team_label.lower()}.png"), bbox_inches='tight')
        plt.close()

    # Csapatszintű összehasonlító statisztikák grafikonja
    def plot_team_comparison(team1_data, team2_data):
        if not team1_data or not team2_data:
            return

        # Átlagok és összegzések
        team1_avg_speed = np.mean([v[0] for v in team1_data.values()])
        team2_avg_speed = np.mean([v[0] for v in team2_data.values()])
        team1_max_speed = np.mean([v[1] for v in team1_data.values()])
        team2_max_speed = np.mean([v[1] for v in team2_data.values()])
        team1_total_distance = np.sum([v[2] for v in team1_data.values()])
        team2_total_distance = np.sum([v[2] for v in team2_data.values()])
        team1_avg_distance = np.mean([v[2] for v in team1_data.values()])
        team2_avg_distance = np.mean([v[2] for v in team2_data.values()])

        metrics = ['Átlagos sebesség', 'Átlagos max sebesség', 'Átlagos megtett távolság', 'Összes megtett távolság']
        team1_values = [team1_avg_speed, team1_max_speed, team1_avg_distance, team1_total_distance]
        team2_values = [team2_avg_speed, team2_max_speed, team2_avg_distance, team2_total_distance]

        x = np.arange(len(metrics))
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(x - 0.1, team1_values, label='Team1', color=team1_color_rgb)
        ax.scatter(x + 0.1, team2_values, label='Team2', color=team2_color_rgb)

        for i, val in enumerate(team1_values):
            unit = 'km/h' if i < 2 else 'm'
            ax.text(x[i] - 0.1, val + 15, f"{val:.1f} {unit}", ha='center', color=team1_color_rgb, fontsize=8)
        for i, val in enumerate(team2_values):
            unit = 'km/h' if i < 2 else 'm'
            ax.text(x[i] + 0.1, val + 15, f"{val:.1f} {unit}", ha='center', color=team2_color_rgb, fontsize=8)

        ax.set_ylabel('Érték')
        ax.set_title('Csapatszintű statisztikák összehasonlítása')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "activity_team_comparison_summary.png"), bbox_inches='tight')
        plt.close()

    # Mindhárom grafikon generálása
    plot_team_stats(player_stats_team1, "Team1")
    plot_team_stats(player_stats_team2, "Team2")
    plot_team_comparison(player_stats_team1, player_stats_team2)
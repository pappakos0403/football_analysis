import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D

def plot_players_per_half_graph(player_coordinates_list, 
                                track_id_to_team, 
                                field_sides, 
                                team1_color, 
                                team2_color, 
                                fps, 
                                output_dir="statistics"):
    
    num_frames = len(player_coordinates_list)

    # Játékosszámok listái
    team1_half_team1, team1_half_team2 = [], []
    team2_half_team1, team2_half_team2 = [], []

    for frame_coords in player_coordinates_list:
        t1h_t1 = t1h_t2 = t2h_t1 = t2h_t2 = 0
        for player_id, (x, _) in frame_coords.items():
            team = track_id_to_team.get(player_id)
            if team not in (1, 2):
                continue
            # Meghatározzuk, hogy az adott játékos a Team1 térfelén van-e
            is_on_team1_half = x < 52.5 if field_sides[1] == "left" else x > 52.5
            if is_on_team1_half:
                if team == 1: t1h_t1 += 1
                else: t1h_t2 += 1
            else:
                if team == 1: t2h_t1 += 1
                else: t2h_t2 += 1
        # Eredmények elmentése a listákba
        team1_half_team1.append(t1h_t1)
        team1_half_team2.append(t1h_t2)
        team2_half_team1.append(t2h_t1)
        team2_half_team2.append(t2h_t2)

    # Időbélyegek létrehozása
    timestamps = np.linspace(0, num_frames / fps, 20)
    time_labels = [f"{int(t//60):02d}:{int(t%60):02d}" for t in timestamps]
    frame_indices = [min(int(t * fps), num_frames - 1) for t in timestamps]

    # Színek átalakítása megfelelő formátumba
    t1_color = tuple(c / 255 for c in team1_color)
    t2_color = tuple(c / 255 for c in team2_color)

    # Kiválasztott frame-ekre szűkítés
    t1h_t1_vals = [team1_half_team1[i] for i in frame_indices]
    t1h_t2_vals = [team1_half_team2[i] for i in frame_indices]
    t2h_t1_vals = [team2_half_team1[i] for i in frame_indices]
    t2h_t2_vals = [team2_half_team2[i] for i in frame_indices]

    # Stílus beállítása és ábra előkészítése
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Főcím a grafikon fölé
    fig.suptitle("Játékosok számának változása mindkét térfélen", fontsize=17, fontweight='bold')

    # --- Team 1 térfél grafikon ---
    axs[0].plot(time_labels, t1h_t1_vals, color=t1_color, marker='o', linewidth=2, label="Team 1 játékosai")
    axs[0].plot(time_labels, t1h_t2_vals, color=t2_color, marker='o', linewidth=2, label="Team 2 játékosai")
    axs[0].set_title("Team 1 térfél", fontsize=14, fontweight='bold')
    axs[0].set_xlabel("Idő (perc:másodperc)")
    axs[0].set_ylabel("Játékosok száma")
    axs[0].set_yticklabels([])
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].grid(True)

    # Számok kiírása minden ponthoz
    for i in range(len(time_labels)):
        axs[0].text(i, t1h_t1_vals[i] + 0.4, str(t1h_t1_vals[i]), color='black', ha='center', fontsize=10, fontweight='bold')
        axs[0].text(i, t1h_t2_vals[i] + 0.4, str(t1h_t2_vals[i]), color='black', ha='center', fontsize=10, fontweight='bold')

    # --- Team 2 térfél grafikon ---
    axs[1].plot(time_labels, t2h_t1_vals, color=t1_color, marker='o', linewidth=2, label="Team 1 játékosai")
    axs[1].plot(time_labels, t2h_t2_vals, color=t2_color, marker='o', linewidth=2, label="Team 2 játékosai")
    axs[1].set_title("Team 2 térfél", fontsize=14, fontweight='bold')
    axs[1].set_xlabel("Idő (perc:másodperc)")
    axs[1].set_ylabel("Játékosok száma")
    axs[1].set_yticklabels([])
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].grid(True)

    # Számok kiírása a pontok fölé
    for i in range(len(time_labels)):
        axs[1].text(i, t2h_t1_vals[i] + 0.4, str(t2h_t1_vals[i]), color='black', ha='center', fontsize=10, fontweight='bold')
        axs[1].text(i, t2h_t2_vals[i] + 0.4, str(t2h_t2_vals[i]), color='black', ha='center', fontsize=10, fontweight='bold')

    # --- Jelmagyarázat létrehozása és középre helyezése ---
    legend_elements = [
        Line2D([0], [0], color=t1_color, lw=2, label='Team 1 játékosai'),
        Line2D([0], [0], color=t2_color, lw=2, label='Team 2 játékosai')
    ]

    fig.legend(
        handles=legend_elements,
        loc='upper center',         
        ncol=2,                     
        fontsize=12,
        frameon=False,              
        bbox_to_anchor=(0.5, 0.94)  
    )

    # Mentés fájlba
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "players_per_half_timed_graph.png")
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(output_path, dpi=150)
    plt.close()
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class OffsideDetector:
    def __init__(self, player_coordinates, ball_coordinates, closest_player_ids_filtered, track_id_to_team, field_sides, flag_path):
        # Bemeneti adatok tárolása
        self.player_coordinates = player_coordinates
        self.ball_coordinates = ball_coordinates
        self.closest_player_ids_filtered = closest_player_ids_filtered
        self.track_id_to_team = track_id_to_team
        self.field_sides = field_sides

        # Zászló betöltése
        self.flag_image = cv2.imread(flag_path, cv2.IMREAD_UNCHANGED)
        if self.flag_image is None:
            raise FileNotFoundError(f"Nem található a zászlókép: {flag_path}")

        # Lesen lévő játékosok listája
        self.offsides_log = []
        self.offsides_stats = defaultdict(int)  # játékosID -> lesen töltött frame-ek száma

    def detect_offsides_per_frame(self):
        # Minden frame-en végigmegyünk
        offsides_per_frame = {}

        for frame_num, player_coords in enumerate(self.player_coordinates):
            # Labdabirtokos lekérése
            ball_owner_id, ball_owner_team = self.closest_player_ids_filtered.get(frame_num, (None, None))
            if ball_owner_team is None:
                continue  # nincs birtokos, nincs les

            # Ellenfél csapata
            opponent_team = 2 if ball_owner_team == 1 else 1

            # Ellenfél játékosainak x koordinátái
            opponent_xs = [coord[0] for pid, coord in player_coords.items() if self.track_id_to_team.get(pid) == opponent_team]
            if len(opponent_xs) < 1:
                continue  # nincs ellenfél játékos a frame-ben

            # Ellenfél leghátrébb levő játékosa
            if self.field_sides[ball_owner_team] == "left":
                last_defender_x = max(opponent_xs)
            else:
                last_defender_x = min(opponent_xs)

            # Labda x koordinátája
            ball_x = self.ball_coordinates[frame_num][0] if frame_num < len(self.ball_coordinates) else None

            # Lesen lévők keresése
            offsides = []
            for pid, (x, _) in player_coords.items():
                if self.track_id_to_team.get(pid) != ball_owner_team:
                    continue
                if pid == ball_owner_id:
                    continue

                if self.field_sides[ball_owner_team] == "left":
                    if x > last_defender_x and (ball_x is None or x > ball_x):
                        offsides.append(pid)
                else:
                    if x < last_defender_x and (ball_x is None or x < ball_x):
                        offsides.append(pid)

            if offsides:
                offsides_per_frame[frame_num] = (offsides, ball_owner_team)
                self.offsides_log.append(f"Frame: {frame_num} Lesen lévő játékos(ok): {','.join(map(str, offsides))} Csapat: {ball_owner_team}")
                for pid in offsides:
                    self.offsides_stats[pid] += 1

        return offsides_per_frame

    def draw_offside_flags(self, frames, offsides_per_frame, tracks):
        # Bejárjuk a frame-eket
        for frame_num, frame in enumerate(frames):
            if frame_num not in offsides_per_frame:
                continue

            lesen_levok, _ = offsides_per_frame[frame_num]
            for pid in lesen_levok:
                player_data = tracks["players"][frame_num].get(pid)
                if not player_data:
                    continue
                x1, y1, x2, y2 = map(int, player_data["bbox"])
                center_x = (x1 + x2) // 2
                top_y = y1 - 55

                fh, fw = self.flag_image.shape[:2]

                for c in range(3):
                    for i in range(fh):
                        for j in range(fw):
                            if self.flag_image[i, j, 3] > 0:
                                y, x = top_y + i, center_x - fw // 2 + j
                                if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                                    frame[y, x, c] = self.flag_image[i, j, c]

        return frames

    def plot_top5_offsides(self, fps, output_dir, team1_color_rgb=(0.0, 0.0, 1.0), team2_color_rgb=(1.0, 0.5, 0.0)):
        # Top5 játékos kiválasztása lesen töltött frame alapján
        top5 = sorted(self.offsides_stats.items(), key=lambda x: x[1], reverse=True)[:5]

        # Ábra előkészítése
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Top 5 játékos lesen töltött idő alapján", fontsize=16, fontweight="bold")

        y_labels = []
        y_colors = []
        times = []
        time_labels = []

        for idx, (track_id, frames) in enumerate(top5, 1):
            total_seconds = frames / fps
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            time_str = f"{minutes:02d}:{seconds:02d}.{int((total_seconds % 1)*100):02d}"

            team_id = self.track_id_to_team.get(track_id, 0)
            color = team1_color_rgb if team_id == 1 else team2_color_rgb

            y_labels.append(f"$\\bf{{{idx}.}}$ Player {track_id}")
            y_colors.append(color)
            times.append(total_seconds)
            time_labels.append(time_str)

        y_pos = np.arange(len(top5))
        bars = ax.barh(y_pos, times, color=y_colors)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()
        ax.set_xlabel("Lesen töltött idő")
        ax.grid(True, axis="x")

        max_x = max(times) if times else 1
        label_x_pos = max_x + 1.0

        for i, (label, bar) in enumerate(zip(time_labels, bars)):
            ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2,
                    label, va='center', ha='left', fontsize=10)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "top5_offside_players.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
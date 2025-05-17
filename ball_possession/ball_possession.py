import numpy as np
from utils import get_center_of_bbox, TeamAssigner
import cv2

class BallPossession:
    def __init__(self, distance_threshold = 60):
        # Labdabirtoklás számlálók inicializálása
        self.team1_possession = 0
        self.team2_possession = 0
        self.total_possession_frames = 0

        # Szűrők a legközelebbi játékos meghatárosához
        self.distance_threshold = distance_threshold # távolság a labdától (pixelben)
        self.streak_threshold = 3
        self.closest_player_streaks = {}
    
    def player_on_the_ball(self, players, ball_bbox, frame_width, frame_height):
        # Labda középpontjának meghatározása
        ball_x, ball_y = get_center_of_bbox(ball_bbox)

        # Minimális bbox méretek a hibás detektálások kiszűrésére
        min_bbox_width = 10
        min_bbox_height = 10

        # Szükséges változók
        min_distance = float('inf')
        closest_player_id = None

        for track_id, player in players.items():
            x1, y1, x2, y2 = player["bbox"]

            # Bbox méreteinek ellenőrzése
            if (x2 - x1) < min_bbox_width or (y2 - y1) < min_bbox_height:
                continue
            
            # Bbox a pályán belül van-e
            if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
                continue

            left_foot = (x1, y2) # bbox bal alsó sarka
            right_foot = (x2, y2) # bbox jobb alsó sarka

            left_distance = np.linalg.norm(np.array(left_foot) - np.array((ball_x, ball_y))) # bal láb távolsága a labdától
            right_distance = np.linalg.norm(np.array(right_foot) - np.array((ball_x, ball_y))) # jobb láb távolsága a labdától

            # Legrövidebb távolság kiszámítása a játékos lábai és a labda között
            distance = min(left_distance, right_distance)

            # Labdához legközelebbi játékos meghatározása
            if distance < min_distance:
                closest_player_id = track_id
                min_distance = distance

        # Ellenőrzés, hogy a távolsághatáron belül van-e a legközelebbi játékos a labdához
        if min_distance <= self.distance_threshold:
            # Streakek ellenőrzése
            if closest_player_id not in self.closest_player_streaks:
                self.closest_player_streaks[closest_player_id] = 1
            else:
                self.closest_player_streaks[closest_player_id] += 1

            # Ha a streak elérte a határt, akkor visszaadjuk a legközelebbi játékost
            if self.closest_player_streaks[closest_player_id] >= self.streak_threshold:
                return closest_player_id
        
        else:
            # Ha már nem a legközelebbi játékos, akkor töröljük a játékos azonosítóját a streakek közül
            if closest_player_id in self.closest_player_streaks:
                del self.closest_player_streaks[closest_player_id]

        return None
    
    def measure_and_draw_possession(self, frames, closest_player_ids_filtered):
        # Végigmegyünk minden frame-en
        for frame_num, frame in enumerate(frames):
            annotated_frame = frame.copy()

            # Szűrt listából kiszedjük az aktuális legközelebbi játékost
            if frame_num in closest_player_ids_filtered:
                player_id, team_id = closest_player_ids_filtered[frame_num]

                # Csak akkor dolgozunk vele, ha nem None
                if player_id is not None and team_id is not None:
                    # Számlálók frissítése a csapat azonosító alapján
                    if team_id == 1:
                        self.team1_possession += 1
                    elif team_id == 2:
                        self.team2_possession += 1

                    self.total_possession_frames += 1

            # Labdabirtoklás megjelenítése minden frame-en
            overlay = annotated_frame.copy()
            h, w, _ = frame.shape
            cv2.rectangle(overlay, (w - 300, 0), (w, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

            # Százalékok számítása
            if self.total_possession_frames > 0:
                team1_pct = 100 * self.team1_possession / self.total_possession_frames
                team2_pct = 100 * self.team2_possession / self.total_possession_frames
            else:
                team1_pct = team2_pct = 0

            # Százalékos kiírás megjelenítése a jobb felső sarokban
            cv2.putText(annotated_frame, f"Team1: {team1_pct:.1f} %", (w - 290, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(annotated_frame, f"Team2: {team2_pct:.1f} %", (w - 290, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # Frissítjük a frame-et
            frames[frame_num] = annotated_frame

        return frames
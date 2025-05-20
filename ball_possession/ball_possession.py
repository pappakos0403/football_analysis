import numpy as np
from utils import get_center_of_bbox, TeamAssigner
import cv2

class BallPossession:
    def __init__(self):
        # Labdabirtoklás számlálók inicializálása
        self.team1_possession = 0
        self.team2_possession = 0
        self.total_possession_frames = 0

        # Szűrők a legközelebbi játékos meghatárosához
        self.distance_threshold = 1.8 # méterben
        self.streak_threshold = 2 # frame-ben
        self.closest_player_streaks = {} # legközelebbi játékos streakjei
    
    def player_on_the_ball(self, pitch_coordinates, ball_coordinates):
        
        # Labda koordinátáinak kiszedése
        ball_x, ball_y = ball_coordinates

        # Szükséges változók
        min_distance = float('inf')
        closest_player_id = None

        # Játékosok koordinátáinak vizsgálata
        for track_id, player_coords in pitch_coordinates.items():
            if track_id == 1:  # labdát kihagyjuk, csak játékosokat vizsgálunk
                continue
            
            player_x, player_y = player_coords
            
            # Távolság számítása méterben (nem pixelben!)
            distance = np.linalg.norm(np.array((player_x, player_y)) - np.array((ball_x, ball_y)))
            
            # Ha ez a legkisebb távolság eddig, akkor frissítjük
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
    
    def measure_and_draw_possession(self, frames, closest_player_ids):
        # Végigmegyünk minden frame-en
        for frame_num, frame in enumerate(frames):
            annotated_frame = frame.copy()

            # Szűrt listából kiszedjük az aktuális legközelebbi játékost
            if frame_num in closest_player_ids:
                player_id, team_id = closest_player_ids[frame_num]

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
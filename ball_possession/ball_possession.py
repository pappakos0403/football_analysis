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
        self.prev_possession_id = None
        self.possession_streak = 0
        self.streak_threshold = 4
    
    def player_on_the_ball(self, players, ball_bbox):
        # Labda középpontjának meghatározása
        ball_x, ball_y = get_center_of_bbox(ball_bbox)

        # Szükséges változók
        min_distance = float('inf')
        closest_player_id = None

        for track_id, player in players.items():
            x1, y1, x2, y2 = player["bbox"]
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
            return closest_player_id
        
        return None
    
    # Labdabirtoklás számítása és megjelenítése
    def measure_and_draw_possession(self, frames, tracks, tracker, team_assigner, team1_color, team2_color, 
                                    goalkeeper_ids, pitch_coordinates_list, field_sides):
        for frame_num, frame in enumerate(frames):
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            annotated_frame = frame

            # Legközelebbi játékos meghatározása a labdához
            if 1 in ball_dict:
                ball_bbox = [int(v) for v in ball_dict[1]["bbox"]]
                closest_player_id = self.player_on_the_ball(player_dict, ball_bbox)

                # Kizárjuk, ha a labda túl magasan van a játékoshoz képest (valószínűleg levegőben van)
                if closest_player_id and closest_player_id in player_dict:
                    ball_y1 = ball_bbox[1]
                    player_y2 = player_dict[closest_player_id]["bbox"][3]
                    if ball_y1 < player_y2 - 50:  # Ha a labda túl magasan van, akkor figyelmen kívül hagyjuk
                        closest_player_id = None

                # Megnézzük, hány framen keresztül birtokolja a játékos a labdát
                if closest_player_id is not None:
                    if closest_player_id == self.prev_possession_id:
                        self.possession_streak += 1
                    else:
                        self.prev_possession_id = closest_player_id
                        self.possession_streak = 1
                else:
                    self.prev_possession_id = None
                    self.possession_streak = 0

                # Ha átment a szűrőkon, akkor csapatot rendelünk a labdát birtokló játékoshoz
                if self.possession_streak >= self.streak_threshold:
                    if closest_player_id and closest_player_id in player_dict:
                        bbox = player_dict[closest_player_id]["bbox"]
                        annotated_frame = tracker.draw_triangle(annotated_frame, bbox, (0, 0, 255))
                    # Ha a legközelebbi játékos a labdán belül van, akkor csapatot rendelünk hozzá
                    if closest_player_id and closest_player_id in player_dict:

                        # Kapusok kezelése
                        if closest_player_id in goalkeeper_ids:
                            coords = pitch_coordinates_list[frame_num]
                            if closest_player_id in coords:
                                x_coord = coords[closest_player_id][0]
                                if field_sides[1] == "left":
                                    team_number = 1 if x_coord < 52.5 else 2
                                else:
                                    team_number = 1 if x_coord > 52.5 else 2
                        # Mezőnyjátékosok kezelése
                        else:
                            bbox = player_dict[closest_player_id]["bbox"]
                            upper_body = team_assigner.get_upper_body_image(frame, bbox)
                            color = team_assigner.get_player_color(upper_body)
                            team_number = team_assigner.get_player_to_team(color, team1_color, team2_color)

                        # Számlálók frissítése
                        if team_number == 1:
                            self.team1_possession += 1
                        elif team_number == 2:
                            self.team2_possession += 1

                        self.total_possession_frames += 1

            # Labdabirtoklás megjelnítése minden frame-en
            overlay = annotated_frame.copy()
            h, w, _ = frame.shape
            cv2.rectangle(overlay, (w - 300, 0), (w, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

            if self.total_possession_frames > 0:
                team1_pct = 100 * self.team1_possession / self.total_possession_frames
                team2_pct = 100 * self.team2_possession / self.total_possession_frames
            else:
                team1_pct = team2_pct = 0

            cv2.putText(annotated_frame, f"Team1: {team1_pct:.1f} %", (w - 290, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(annotated_frame, f"Team2: {team2_pct:.1f} %", (w - 290, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            frames[frame_num] = annotated_frame

        return frames
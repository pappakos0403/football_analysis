import numpy as np
from utils import get_center_of_bbox

class BallPossession:
    def __init__(self, distance_threshold = 60):
        # Maximum távolság a labdától
        self.distance_threshold = distance_threshold
    
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
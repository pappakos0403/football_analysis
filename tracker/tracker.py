from ultralytics import YOLO
from utils.bbox_utils import get_center_of_bbox, get_bbox_width
from utils.team_assigner_utils import TeamAssigner
from ball_possession import BallPossession
from pitch_config import FootballPitchConfiguration
import pickle
import os
import cv2
import numpy as np
import pandas as pd
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        # Tracker modellhez szükséges inicializáció
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

        # Csapatok színének meghatározásához szükséges változók
        self.teamAssigner = TeamAssigner()
        self.team1_color = None
        self.team2_color = None
        self.threshold = 70

        # Labdabirtoklás méréséhez szükséges változók
        self.possession = BallPossession()
        self.team1_possession = 0
        self.team2_possession = 0
        self.total_possession_frames = 0

        # Kapus azonosítók tárolása
        self.goalkeeper_ids = set()

        # Kulcspontokhoz szükséges változók
        self.pitch_edges = FootballPitchConfiguration().edges

    def draw_ellipse(self, frame, bbox, color, track_id = None):
        # Alsó koordináta a bbox alapján
        y2 = int(bbox[3])  

        # Középpont és szélesség kiszámítása a bbox segítségével
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Elipszis rajzolása a játékos alá
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Track_id megjelenítése a játékos alatt
        if track_id is not None:
            # Text generálása
            text = str(track_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_w, text_h = text_size

            # Téglalap pozicionálása
            rect_top_left = (x_center - text_w // 2, y2 - text_h - 10)
            rect_bottom_right = (x_center + text_w // 2 + 2 , y2 - 10)

            # Téglalap kitöltése fehér színnel
            cv2.rectangle(frame, rect_top_left, rect_bottom_right, color=(255, 255, 255), thickness=cv2.FILLED)
            cv2.putText(frame, text, (x_center - text_w // 2, y2 - 10), font, font_scale, (0, 0, 0), thickness)
        
        return frame

    def draw_triangle(self, frame, bbox, color):
        # Felső koordináta a bbox alapján
        x1, y1, x2, y2 = map(int, bbox)

        # Háromszög mérete
        center_x = (x1 + x2) // 2
        triangle_width = 30

        # Háromszög csúcsainak meghatározása
        triangle_points = np.array([
            [center_x, y1 - 15],  # Csúcs
            [center_x - (triangle_width // 2), y1 - 30],         # Bal Felső sarok
            [center_x + (triangle_width // 2), y1 - 30]          # Jobb Felső sarok
        ], np.int32)

        # Háromszög kitöltése
        cv2.fillPoly(frame, [triangle_points], color)

        # Háromszög kontúrjának rajzolása
        cv2.polylines(frame, [triangle_points], isClosed=True, color=(0, 0, 0), thickness=2)

        return frame
    
    # Labda detektálása minden framen
    def interpolate_ball(self, positions):

        # Ha a labda nincs detektálva, akkor egy üres szótár lesz
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in positions]

        # DataFrame létrehozása
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        # Üres értékek interpolációja
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # Visszalakítás numpy formába
        interpolated_ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return interpolated_ball_positions

    def detect_video(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                annotated_frames = pickle.load(f)
            return annotated_frames
        
        detections = []

        for frame_num, frame in enumerate(frames):
            detection = self.model(frame)[0]
            detections.append(detection)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        # Detektálás az összes képkockán
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {value:key for key, value in cls_names.items()}

            # Objektumok követése supervision-nel
            detection_supervision = sv.Detections.from_ultralytics(detection)
                    
            tracked_objects = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in tracked_objects:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_names[cls_id] == "goalkeeper":
                    self.goalkeeper_ids.add(track_id)
                    cls_id = cls_names_inv['player']

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        # Labda pozíciók interpolálása
        tracks["ball"] = self.interpolate_ball(tracks["ball"])

        return tracks


    def annotations(self, frames, tracks, camera_movements=None, keypoints_list=None, pitch_coordinates_list=None):

        annotated_frames = []

        # track_id gyorsítótár
        self.track_id_to_team = {}

        for frame_num, frame in enumerate(frames):
            annotated_frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Csapatszín beállítása az első frame-en
            if frame_num == 0:
                for _, player in player_dict.items():
                    # Első csapat színének meghatározása
                    if self.team1_color is None:
                        upper_body_image = self.teamAssigner.get_upper_body_image(frame, player["bbox"])
                        self.team1_color = self.teamAssigner.get_player_color(upper_body_image)
                        continue
                    
                    # Második csapat színének meghatározása
                    upper_body_image = self.teamAssigner.get_upper_body_image(frame, player["bbox"])
                    player_color = self.teamAssigner.get_player_color(upper_body_image)

                    # Színkülönbség számítása
                    color_diff = np.linalg.norm(np.array(self.team1_color) - np.array(player_color))
                    if color_diff > self.threshold:
                        self.team2_color = player_color
                        break

            # Játékosok:
            for track_id, player in player_dict.items():
                # Csak a mezőnyjátékosokat vizsgáljuk
                if track_id in self.goalkeeper_ids:
                    continue
                # Játékos színének meghatározása
                if track_id not in self.track_id_to_team:
                    upper_body_image = self.teamAssigner.get_upper_body_image(frame, player["bbox"])
                    apperance_feature = self.teamAssigner.get_player_color(upper_body_image)
                    team_color_num = self.teamAssigner.get_player_to_team(apperance_feature, self.team1_color, self.team2_color)
                    self.track_id_to_team[track_id] = team_color_num
                else:
                    team_color_num = self.track_id_to_team[track_id]

                # Elipszis rajzolása a játékosok alá
                if team_color_num == 1:
                    annotated_frame = self.draw_ellipse(annotated_frame, player["bbox"], self.team1_color, track_id=track_id)
                elif team_color_num == 2:
                    annotated_frame = self.draw_ellipse(annotated_frame, player["bbox"], self.team2_color, track_id=track_id)

            # Játékvezetők:
            for track_id, referee in referee_dict.items():
                # Sárga elipszis rajzolása a játékvezető alá
                referee_color = (0, 255, 255)
                annotated_frame = self.draw_ellipse(annotated_frame, referee["bbox"], referee_color)

            # Labda:
            if 1 in ball_dict:
                ball_bbox = [int(v) for v in ball_dict[1]["bbox"]]
                # Labda fölé zöld háromszög rajzolása
                annotated_frame = self.draw_triangle(annotated_frame, ball_bbox, (0, 255, 0))

                # Legközelebbi játékos meghatározása a labdához
                closest_player_id = self.possession.player_on_the_ball(player_dict, ball_bbox)

                if closest_player_id is not None:
                    closest_player_bbox = player_dict[closest_player_id]["bbox"]
                    
            # Kulcspontok és vonalak kirajzolása, ha van keypoint adat
            if keypoints_list and frame_num < len(keypoints_list):
                pts = keypoints_list[frame_num]
                if pts is not None and pts.size > 0:
                    # Kulcspontok kirajzolása
                    for pt in pts:
                        cv2.circle(annotated_frame, (int(pt[0]), int(pt[1])), radius=4, color=(255, 0, 0), thickness=-1)
                    # Kulcspontok összekötése -> pályavonalak rajzolása
                    for edge in self.pitch_edges:
                        idx1, idx2 = edge[0]-1, edge[1]-1
                        if idx1 < len(pts) and idx2 < len(pts):
                            pt1 = (int(pts[idx1][0]), int(pts[idx1][1]))
                            pt2 = (int(pts[idx2][0]), int(pts[idx2][1]))
                            cv2.line(annotated_frame, pt1, pt2, color=(255, 255, 0), thickness=2)

            # Játékosok pályakoordinátáinak kiírása
            if pitch_coordinates_list and frame_num < len(pitch_coordinates_list):
                coords = pitch_coordinates_list[frame_num]
                for track_id, player in tracks["players"][frame_num].items():
                    if track_id in coords:
                        x1,y1,x2,y2 = player["bbox"]
                        x_center = int((x1+x2)/2)
                        y_bottom = int(y2)
                        text = f"x: {coords[track_id][0]:.1f}m y: {coords[track_id][1]:.1f}m"
                        cv2.putText(annotated_frame, text, (x_center-70, y_bottom+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            
            annotated_frames.append(annotated_frame)

        return annotated_frames
    
    # Kapusokat külön annotáljuk
    def goalkeeper_annotations(self, annotated_frames, tracks, frames, pitch_coordinates_list, field_sides):
        for frame_num, frame in enumerate(frames):
            annotated_frame = annotated_frames[frame_num]
            player_dict = tracks["players"][frame_num]
            coords = pitch_coordinates_list[frame_num] if pitch_coordinates_list and frame_num < len(pitch_coordinates_list) else {}

            # Csak a kapusokat vizsgáljuk
            for track_id, player in player_dict.items():
                if track_id not in self.goalkeeper_ids:
                    continue

                # Térfél szerint besoroljuk a kapust a megfelelő csapatba
                if track_id in coords:
                    x_coord = coords[track_id][0]
                    if field_sides[1] == "left":
                        team_number = 1 if x_coord < 52.5 else 2
                    else:
                        team_number = 1 if x_coord > 52.5 else 2

                    # Kapus annotációk rajzolása
                    if team_number == 1:
                        annotated_frame = self.draw_ellipse(annotated_frame, player["bbox"], self.team1_color, track_id=track_id)
                    else:
                        annotated_frame = self.draw_ellipse(annotated_frame, player["bbox"], self.team2_color, track_id=track_id)

            # Annotált képkockák frissítése
            annotated_frames[frame_num] = annotated_frame

        return annotated_frames
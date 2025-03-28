from ultralytics import YOLO
from utils.video_utils import load_video, generate_output_video
from utils.bbox_utils import get_center_of_bbox, get_bbox_width
from utils.team_assigner_utils import TeamAssigner
import cv2
import numpy as np
import pandas as pd
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

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

    def draw_triangle(self, frame, bbox):
        # Felső koordináta a bbox alapján
        x1, y1, x2, y2 = map(int, bbox)

        # Háromszög csúcsainak meghatározása
        triangle_points = np.array([
            [(x1 + x2) // 2, y1 + 20],  # Csúcs a labda felett
            [x1, y1],                   # Bal alsó sarok
            [x2, y1]                    # Jobb alsó sarok
        ], np.int32)

        # Háromszög kitöltése
        cv2.fillPoly(frame, [triangle_points], (0, 255, 0))

        # Háromszög kontúrjának rajzolása
        cv2.polylines(frame, [triangle_points], isClosed=True, color=(0, 0, 0), thickness=2)

        return frame

    def interpolate_ball_positions(self, ball_positions):
            ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
            df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

            # Hiányzó értékek interpolálása
            df_ball_positions = df_ball_positions.interpolate()
            df_ball_positions = df_ball_positions.bfill()

            ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

            return ball_positions

    def detect_video(self, frames, fps, width, height):
        # Csapatok színéhez szüksges változók
        team1_color = None
        team2_color = None
        threshold = 70
        
        annotated_frames = []

        # Detektálás az összes képkockán
        for frame_num, frame in enumerate(frames):
            annotated_frame = frame.copy()

            # YOLO detekció
            results = self.model(frame)
            detection_supervision = sv.Detections.from_ultralytics(results[0])
            tracked_objects = self.tracker.update_with_detections(detection_supervision)

            # Az eredményekből a bounding boxok kinyerése
            boxes = results[0].boxes

            # Kapus átállítása játékosra
            # boxes.data tensor oszlopainál a cls átállítása 2-ről 1-re (kapus -> játékos)
            boxes_data = boxes.data.clone()
            boxes_data.data[boxes.data[:, 5] == 1, 5] = 2
            boxes.data = boxes_data

            # boxes.data numpy tömbbé alakítása
            boxes_np = boxes.data.cpu().numpy() if hasattr(boxes.data, "cpu") else boxes.data

            # Külön kezeljük a játékosokat, a játékvezetőket és a labdát
            detected_objects = {
                "players": [], # cls: 2
                "referees": [], # cls: 3
                "ball": [] # cls: 0
            }

            # Bounding boxok feldolgozása
            for box in boxes_np:
                x1, y1, x2, y2, conf, cls = box
                cls = int(cls)

                # Játékosok
                if cls == 2 and conf > 0.75:
                    detected_objects["players"].append([x1, y1, x2, y2])

                # Játékvezetők
                elif cls == 3:
                    detected_objects["referees"].append([x1, y1, x2, y2])

                # Labda
                elif cls == 0:
                    detected_objects["ball"].append([x1, y1, x2, y2])

            # Játékos színének meghatározása
            teamAssigner = TeamAssigner()

            if frame_num == 0:
                for id, player in enumerate(detected_objects["players"]):

                    # Első csapat színének meghatározása
                    if team1_color is None:
                        upper_body_image = teamAssigner.get_upper_body_image(frame, player, id)
                        team1_color = teamAssigner.get_player_color(upper_body_image, id)
                        continue

                    # Második csapat színének meghatározása
                    upper_body_image = teamAssigner.get_upper_body_image(frame, player, id)
                    player_color = teamAssigner.get_player_color(upper_body_image, id)

                    # Színkülönbség kiszámítása Euklideszi távolsággal
                    color_diff = np.linalg.norm(np.array(team1_color) - np.array(player_color))
                    if color_diff > threshold:
                        team2_color = player_color
                        break   
            
            # Elipszis és ID megjelenítése minden követett játékoson és játékvezetőn
            for bbox in detected_objects["referees"]:
                if bbox in detected_objects["referees"]:
                    # Elipszis rajzolása sárga színnel a játékvezetők számára
                    referee_color = (0, 255, 255)
                    annotated_frame = self.draw_ellipse(annotated_frame, bbox, referee_color, track_id=None)

            for track_id, bbox in enumerate(detected_objects["players"]):
                # Csapatszín meghatározása a TeamAssigner-rel
                upper_body_image = teamAssigner.get_upper_body_image(frame, bbox, track_id)
                apperance_feature = teamAssigner.get_player_color(upper_body_image, track_id)
                team_color_num = teamAssigner.get_player_to_team(apperance_feature, team1_color, team2_color, track_id)

                # Elipszis rajzolása a megfelelő színnel
                if team_color_num == 1:
                    annotated_frame = self.draw_ellipse(annotated_frame, bbox, team1_color, track_id=track_id)
                elif team_color_num == 2:
                    annotated_frame = self.draw_ellipse(annotated_frame, bbox, team2_color, track_id=track_id)

            # Annotált képkocka hozzáadása a listához
            annotated_frames.append(annotated_frame)
        
        return annotated_frames, fps, width, height
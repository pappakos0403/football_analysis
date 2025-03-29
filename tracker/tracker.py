from ultralytics import YOLO
from utils.bbox_utils import get_center_of_bbox, get_bbox_width
from utils.team_assigner_utils import TeamAssigner
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
            [(x1 + x2) // 2, y1 - 15],  # Csúcs a labda felett
            [x1, y1 - 30],         # Bal Felső sarok
            [x2, y1 - 30]          # Jobb Felső sarok
        ], np.int32)

        # Háromszög kitöltése
        cv2.fillPoly(frame, [triangle_points], (0, 255, 0))

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

    def detect_video(self, frames, fps, width, height):
        
        annotated_frames = []
        raw_ball_positions = []

        # Detektálás az összes képkockán
        for frame_num, frame in enumerate(frames):
            annotated_frame = frame.copy()

            # Objektumok követése supervision-nel
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

            if frame_num == 0:
                for player in detected_objects["players"]:
                    # Első csapat színének meghatározása
                    if self.team1_color is None:
                        upper_body_image = self.teamAssigner.get_upper_body_image(frame, player)
                        self.team1_color = self.teamAssigner.get_player_color(upper_body_image)
                        continue

                    # Második csapat színének meghatározása
                    upper_body_image = self.teamAssigner.get_upper_body_image(frame, player)
                    player_color = self.teamAssigner.get_player_color(upper_body_image)

                    # Színkülönbség kiszámítása Euklideszi távolsággal
                    color_diff = np.linalg.norm(np.array(self.team1_color) - np.array(player_color))
                    if color_diff > self.threshold:
                        self.team2_color = player_color
                        break
            
            # Labda pozíciójának eltárolása
            if detected_objects["ball"]:
                raw_ball_positions.append({1: {"bbox": detected_objects["ball"][0]}})
            else:
                raw_ball_positions.append({})

            # Detektált objektumok vizsgálata
            for detection in tracked_objects:
                
                xyxy = detection[0]
                class_id = detection[3]
                track_id = detection[4]

                bbox = xyxy.tolist()
                
                # Játékosok:
                if class_id == 2:
                    # Csapatszín meghatározása a TeamAssigner-rel
                    upper_body_image = self.teamAssigner.get_upper_body_image(frame, bbox)
                    apperance_feature = self.teamAssigner.get_player_color(upper_body_image)
                    team_color_num = self.teamAssigner.get_player_to_team(apperance_feature, self.team1_color, 
                                                                            self.team2_color)

                    # Elipszis rajzolása a megfelelő színnel
                    if team_color_num == 1:
                        annotated_frame = self.draw_ellipse(annotated_frame, bbox, self.team1_color, track_id=track_id)
                    elif team_color_num == 2:
                        annotated_frame = self.draw_ellipse(annotated_frame, bbox, self.team2_color, track_id=track_id)

                # Játékvezetők:
                elif class_id == 3:
                    referee_color = (0, 255, 255)
                    annotated_frame = self.draw_ellipse(annotated_frame, bbox, referee_color, track_id=None)

            # Labda:
            interpolated_ball_positions = self.interpolate_ball(raw_ball_positions)
            if 1 in interpolated_ball_positions[frame_num]:
                bbox = [int(v) for v in interpolated_ball_positions[frame_num][1]["bbox"]]
                annotated_frame = self.draw_triangle(annotated_frame, bbox)

            # Annotált képkocka hozzáadása a listához
            annotated_frames.append(annotated_frame)
        
        return annotated_frames, fps, width, height
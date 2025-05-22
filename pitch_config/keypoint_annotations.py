import cv2
import numpy as np
from ultralytics import YOLO
from pitch_config import FootballPitchConfiguration
from .view_transformer import ViewTransformer

def process_keypoint_annotations(video_path, keypoint_model_path, players_tracks=None, ball_tracks=None):
    
    # Konfidencia határérték
    confidence_threshold = 0.5
    
    # Videó megnyitása
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Nem sikerült megnyitni a videót!")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Képátméretezéshez szükséges skálázási tényezők a 640×640-es detektáláshoz
    scale_x_det = frame_width / 640.0
    scale_y_det = frame_height / 640.0

    # Pálya konfiguráció betöltése
    pitch_config = FootballPitchConfiguration()
    pitch_vertices_cm = np.array(pitch_config.vertices, dtype=np.float32)
    
    frame_count = 0
    all_keypoints = []       # minden képkocka detektált kulcspontjait tartalmazza
    all_player_coords = []    # minden képkockára (x, y) játékos koordináta
    all_ball_coords = []      # minden képkockára (x, y) labda koordináta
    
    # Keypoint detektáló modell betöltése
    model = YOLO(keypoint_model_path)
    
    while True:
        ret, fullhd_frame = cap.read()
        if not ret:
            break
        
        frame_keypoints = None  # kulcspontok a képkockán
        player_pitch_coords = {} # játékos pályakoordináták (track_id → (x, y))
        homography_valid = False
        
        # Átméretezés a 640x640-es bemenetre
        resized_frame = cv2.resize(fullhd_frame, (640, 640))
        results = model(resized_frame)
        result = results[0]
        
        if hasattr(result, "keypoints") and result.keypoints is not None:
            kp_array = result.keypoints.xy.cpu().numpy()[0]  # alak: (N, 2)
            conf_array = result.keypoints.conf.cpu().numpy()[0]
            # Visszaskálázás FULLHD méretre
            detected_points = np.array([[x * scale_x_det, y * scale_y_det] for x, y in kp_array])
            
            # Csak a megbízható pontok kiválasztása
            valid_filter = conf_array >= confidence_threshold
            if np.sum(valid_filter) >= 4:
                image_points = detected_points[valid_filter]
                pitch_points = pitch_vertices_cm[valid_filter]
                transformer = ViewTransformer(source=pitch_points, target=image_points)
                homography_valid = True
                corrected_points = transformer.transform_points(points=pitch_vertices_cm)
                frame_keypoints = corrected_points
            else:
                frame_keypoints = detected_points
        else:
            print(f"Frame {frame_count}: Nem érhető el keypoint adat!")
            frame_keypoints = np.empty((0, 2))
        
        # Pályakoordináták kiszámítása (játékosok)
        if homography_valid and players_tracks is not None and frame_count < len(players_tracks):
            inverse_transformer = ViewTransformer(source=image_points, target=pitch_points)
            for track_id, player in players_tracks[frame_count].items():
                bbox = player.get("bbox", None)
                if bbox is None or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                x_center = (x1 + x2) / 2
                y_bottom = y2
                player_point = np.array([[x_center, y_bottom]], dtype=np.float32)
                transformed_point = inverse_transformer.transform_points(player_point)
                x_field = transformed_point[0][0] / 100.0  # átváltás centiméterről méterre
                y_field = transformed_point[0][1] / 100.0
                y_field = (pitch_config.width / 100.0) - y_field  # y tengely megfordítása
                player_pitch_coords[track_id] = (x_field, y_field)

        # Pályakoordináták kiszámítása (labda)
        if ball_tracks is not None and frame_count < len(ball_tracks):
            ball_data = ball_tracks[frame_count].get(1, {}).get("bbox", None)
            if ball_data is not None:
                x1, y1, x2, y2 = ball_data
                ball_x = (x1 + x2) / 2
                ball_y = (y1 + y2) / 2
                ball_point = np.array([[ball_x, ball_y]], dtype=np.float32)
                transformed_ball_point = inverse_transformer.transform_points(ball_point)
                ball_x_field = transformed_ball_point[0][0] / 100.0
                ball_y_field = transformed_ball_point[0][1] / 100.0
                ball_y_field = (pitch_config.width / 100.0) - ball_y_field
                ball_pitch_coords = (ball_x_field, ball_y_field)
        
        all_keypoints.append(frame_keypoints)
        all_player_coords.append(player_pitch_coords)
        all_ball_coords.append(ball_pitch_coords)
        frame_count += 1

    cap.release()
    
    return {
        "keypoints": all_keypoints,
        "player_coordinates": all_player_coords,
        "ball_coordinates": all_ball_coords
    }
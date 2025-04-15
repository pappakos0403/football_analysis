import cv2
import numpy as np
from ultralytics import YOLO
from pitch_config import FootballPitchConfiguration
from view_transformer import ViewTransformer
import os
import pickle

# Fájl elérési útvonalak
INPUT_VIDEO_PATH = "input_videos//08fd33_4.mp4"
OUTPUT_VIDEO_PATH = "output_videos//test_keypoints.avi"
MODEL_PATH = "models//best_keypoints.pt"
STUB_PLAYERS_PATH = "stubs//08fd33_4.pkl"  

# Videó beolvasása, paraméterek
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print("Nem sikerült megnyitni a videót!")
    exit()
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Input videó felbontása: {frame_width}x{frame_height}")

# Kimeneti videó létrehozása FULLHD felbontásban
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# Keypoint detektáló modell betöltése
model = YOLO(MODEL_PATH)

# Detektált keypointok skálázási tényezői 640-es -> FULLHD-re
scale_x_det = frame_width / 640.0
scale_y_det = frame_height / 640.0

# Pálya konfiguráció betöltése
pitch_config = FootballPitchConfiguration()
# Eredeti pálya koordináták cm-ben
pitch_vertices_cm = np.array(pitch_config.vertices, dtype=np.float32)
# A pálya vonalainak koordinátái (a pálya szélessége és hossza alapján)
scale_target_x = frame_width / pitch_config.length
scale_target_y = frame_height / pitch_config.width
pitch_vertices_scaled = pitch_vertices_cm.copy()
for i in range(len(pitch_vertices_scaled)):
    x, y = pitch_vertices_scaled[i]
    pitch_vertices_scaled[i] = [x * scale_target_x, y * scale_target_y]

# Keypoint határérték
confidence_threshold = 0.5

# Stub beolvasása
players_tracks = None
if os.path.exists(STUB_PLAYERS_PATH):
    try:
        with open(STUB_PLAYERS_PATH, "rb") as f:
            stub_data = pickle.load(f)
        
        if isinstance(stub_data, dict) and "tracks" in stub_data:
            players_tracks = stub_data["tracks"]["players"]
        else:
            players_tracks = stub_data
        print("Játékosdetekciós stub betöltve.")
    except Exception as e:
        print("Hiba a játékosdetekciós stub betöltésekor:", e)
else:
    print("Játékosdetekciós stub fájl nem található.")

frame_count = 0

while True:
    ret, fullhd_frame = cap.read()
    if not ret:
        break

    annotated_frame = fullhd_frame.copy()

    # Kép átméretezése 640x640-re
    resized_frame = cv2.resize(fullhd_frame, (640, 640))

    # Modell futtatása az átméretezett képen
    results = model(resized_frame)
    result = results[0]

    homography_valid = False  # jelöljük, ha van érvényes homográfia
    if hasattr(result, "keypoints") and result.keypoints is not None:
        kp_array = result.keypoints.xy.cpu().numpy()[0]  # alak: (N, 2)
        conf_array = result.keypoints.conf.cpu().numpy()[0]

        # Pontok visszaskálázása a FULLHD méretre
        detected_points = np.zeros_like(kp_array)
        for i, (x, y) in enumerate(kp_array):
            detected_points[i] = [x * scale_x_det, y * scale_y_det]

        # Csak a megbízható pontok
        valid_filter = conf_array >= confidence_threshold  
        if np.sum(valid_filter) >= 4:
            # Az érvényes képpontok
            image_points = detected_points[valid_filter]
            # A pálya pontjainak koordinátái
            pitch_points = pitch_vertices_cm[valid_filter]

            # Homográfia számítása
            transformer = ViewTransformer(source=pitch_points, target=image_points)
            homography_valid = True

            # Korrigált pontok számítása a pálya koordináták alapján
            corrected_points = transformer.transform_points(points=pitch_vertices_cm)
            for i in range(len(detected_points)):
                if conf_array[i] >= confidence_threshold:
                    corrected_points[i] = detected_points[i]

            # Pontok kirajzolása
            for (x, y) in corrected_points:
                cv2.circle(annotated_frame, (int(x), int(y)), radius=5, color=(230, 216, 173), thickness=-1)
            
            # Pontok összekötése
            for edge in pitch_config.edges:
                i, j = edge
                pt1 = corrected_points[i - 1]  # mivel az élek 1-indexűek
                pt2 = corrected_points[j - 1]
                cv2.line(annotated_frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (139, 0, 0), 2)
    else:
        print(f"Frame {frame_count}: Nem érhető el keypoint adat!")
    
    # Játékosok koordinátájának kiszámítása és kiírása
    if homography_valid:
        inverse_transformer = ViewTransformer(source=image_points, target=pitch_points)
        
        # Koordináták kiírása
        if players_tracks is not None and frame_count < len(players_tracks):
            for track_id, player in players_tracks[frame_count].items():
                bbox = player.get("bbox", None)
                if bbox is None or len(bbox) != 4:
                    continue
                # Lábak pozíciója
                x1, y1, x2, y2 = bbox
                x_center = (x1 + x2) / 2
                y_bottom = y2
                # Készítünk egy (1,1,2) alakú tömböt a transzformációhoz
                player_point = np.array([[x_center, y_bottom]], dtype=np.float32)
                transformed_point = inverse_transformer.transform_points(player_point)
                
                x_field = transformed_point[0][0] / 100.0
                y_field = transformed_point[0][1] / 100.0
                # y koordináta megfordítása
                y_field = (pitch_config.width / 100.0) - y_field

                text = f"x: {x_field:.1f}m y: {y_field:.1f}m"

                # Szöveg kiírása
                pos = (int(x_center - 40), int(y_bottom + 30))
                cv2.putText(annotated_frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    out.write(annotated_frame)
    frame_count += 1

cap.release()
out.release()
print(f"Feldolgozva: {frame_count} képkocka. Kimeneti videó mentve: {OUTPUT_VIDEO_PATH}")

from utils import load_video, generate_output_video
from tracker import Tracker
from camera_movement import CameraMovement
from pitch_config import process_keypoint_annotations
from ultralytics import YOLO
import cv2
import os
import pickle

# Videófájl elérési útvonalai
video_path = "input_videos\\08fd33_4.mp4"
output_video_path = "output_videos\\output_video.avi"
stub_path = "stubs\\08fd33_4.pkl"

# Objektumdetektáló modell elérési útvonala
model_path = "models\\best.pt"

# Keypoint detektáló modell elérési útvonala
keypoint_model_path = "models\\best_keypoints.pt"

# Modell betöltése
tracker = Tracker(model_path)

# Videó betöltése
frames, fps, width, height = load_video(video_path)

# Inicializálás
tracks = None
camera_movements = None
keypoint_data = None

# Stub fájl ellenőrzése
if os.path.exists(stub_path):
    try:
    # Stub betöltése, ha létezik
        with open(stub_path, "rb") as f:
            stub_data = pickle.load(f)

        # Szerkezet
        if (isinstance(stub_data, dict) and 
            "tracks" in stub_data and 
            "camera_movements" in stub_data and 
            "keypoints" in stub_data and
            "player_coordinates" in stub_data):

            tracks = stub_data["tracks"]
            camera_movements = stub_data["camera_movements"]
            keypoint_data = {
                "keypoints": stub_data["keypoints"],
                "player_coordinates": stub_data["player_coordinates"]
            }
            print("Stub fájl betöltve!")
        else:
            print("Hibás .pkl, töröld!")
    except:
        print("Hiba a stub fájl betöltésekor!")

else:
    print("Stub fájl nem található, új generálás indul...")

    # Objektumdetektálás
    tracks = tracker.detect_video(frames, read_from_stub=False, stub_path=None)
    print("Játékosok, játékvezetők és labda detektálva!")

    # Kameramozgás számítása
    camera_estimator = CameraMovement(frames[0])
    camera_movements = camera_estimator.calculate_movement(frames, read_from_stub=False, stub_path=None)
    print("Kameramozgás változása kiszámítva!")

    # Kulcspontok és játékoskoordináták számítása
    keypoint_data = process_keypoint_annotations(players_tracks=tracks["players"])
    print("Kulcspontok detektálva és játékoskoordináták kiszámítva!")

    # Mentés stub fájlba
    stub_data = {
        "tracks": tracks,
        "camera_movements": camera_movements,
        "keypoints": keypoint_data.get("keypoints", []),
        "player_coordinates": keypoint_data.get("player_coordinates", [])
    }
    os.makedirs(os.path.dirname(stub_path), exist_ok=True)
    with open(stub_path, "wb") as f:
        pickle.dump(stub_data, f)
    print("Stub fájl mentve:", stub_path)

# Kameramozgás becslése
camera_estimator = CameraMovement(frames[0])
camera_estimator.adjust_tracks(tracks, camera_movements)

# Pozíció korrekció
camera_estimator.adjust_tracks(tracks, camera_movements)

# Annotálás a videón
annotated_frames = tracker.final_annotations(
    frames, 
    tracks,
    keypoint_data.get("keypoints", []),
    keypoint_data.get("player_coordinates", [])
)
print("Annotálás befejeződött!")

# === Kameramozgás szöveges megjelenítése az annotált videón ===
for i, frame in enumerate(annotated_frames):
    dx, dy = camera_movements[i]
    text1 = f"Kamera elmozdulasa X: {dx:.2f}"
    text2 = f"Kamera elmozdulasa Y: {dy:.2f}"

    # Átlátszó overlay hozzáadása
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Szöveg kiírása
    cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(frame, text2, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

# Output videó generálása
generate_output_video(annotated_frames, output_video_path, fps, width, height)
print("Kimeneti videó mentve:", output_video_path)
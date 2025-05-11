from utils import load_video, generate_output_video, get_majority_team_sides
from tracker import Tracker
from pitch_config import process_keypoint_annotations
from ultralytics import YOLO
from heatmaps import generate_player_heatmaps
from ball_possession import BallPossession
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
            "goalkeeper_ids" in stub_data and 
            "keypoints" in stub_data and
            "player_coordinates" in stub_data):

            tracks = stub_data["tracks"]
            tracker.goalkeeper_ids = stub_data.get("goalkeeper_ids", set())
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

    # Kulcspontok és játékoskoordináták számítása
    keypoint_data = process_keypoint_annotations(players_tracks=tracks["players"])
    print("Kulcspontok detektálva és játékoskoordináták kiszámítva!")

    # Mentés stub fájlba
    stub_data = {
        "tracks": tracks,
        "goalkeeper_ids": tracker.goalkeeper_ids,
        "keypoints": keypoint_data.get("keypoints", []),
        "player_coordinates": keypoint_data.get("player_coordinates", [])
    }
    os.makedirs(os.path.dirname(stub_path), exist_ok=True)
    with open(stub_path, "wb") as f:
        pickle.dump(stub_data, f)
    print("Stub fájl mentve:", stub_path)

# Annotálás a videón
annotated_frames = tracker.annotations(
    frames,
    tracks,
    keypoints_list=keypoint_data.get("keypoints", []),
    pitch_coordinates_list=keypoint_data.get("player_coordinates", [])
)
print("Annotálás befejeződött!")

# Csapatok térfél-hozzárendelésének meghatározása
field_sides = get_majority_team_sides(
    player_coordinates=keypoint_data["player_coordinates"],
    players_tracks=tracks["players"],
    first_frame=frames[0],
    team1_color=tracker.team1_color,
    team2_color=tracker.team2_color,
    team_assigner=tracker.teamAssigner
)
print("Csapatok térfél-hozzárendelése befejeződött!")

# Kapusok annotálása a videón
annotated_frames = tracker.goalkeeper_annotations(
    annotated_frames,
    tracks,
    frames,
    keypoint_data.get("player_coordinates", []),
    field_sides
)
print("Kapusok annotálása befejeződött!")

# Labdabirtoklás számítása és megjelenítése
possession = BallPossession()
annotated_frames = possession.measure_and_draw_possession(
    annotated_frames,
    tracks,
    tracker,
    tracker.teamAssigner,
    tracker.team1_color,
    tracker.team2_color,
    tracker.goalkeeper_ids,
    keypoint_data.get("player_coordinates", []),
    field_sides
)
print("Labdabirtoklás számítása és megjelenítése befejeződött!")

# Hőtérképek generálása
generate_player_heatmaps(keypoint_data.get("player_coordinates", []))
print("Hőtérképek elmentve a heatmaps mappába!")

# Output videó generálása
generate_output_video(annotated_frames, output_video_path, fps, width, height)
print("Kimeneti videó mentve:", output_video_path)

print("Kapus ID-k:", tracker.goalkeeper_ids)
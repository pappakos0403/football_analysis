from utils import load_video, generate_output_video, get_majority_team_sides, closest_player_ids_filter
from tracker import Tracker
from pitch_config import process_keypoint_annotations
from ultralytics import YOLO
from heatmaps import generate_player_heatmaps, generate_ball_heatmap
from ball_possession import BallPossession
from passing_measurement import PassCounter
from player_positions_per_frame import plot_players_per_half_graph
from player_activity import generate_player_activity_summary
from offside_detection import OffsideDetector
import numpy as np
import os
import pickle

# Videófájl elérési útvonalai
filename = "08fd33_4.mp4"
output_filename = f"annotated_{os.path.splitext(filename)[0]}.avi"

video_path = f"input_videos\\{filename}"
stub_path = f"stubs\\{os.path.splitext(filename)[0]}.pkl"
output_video_path = f"output_videos\\{output_filename}"

# Objektumdetektáló modell elérési útvonala
model_path = "models\\best.pt"

# Keypoint detektáló modell elérési útvonala
keypoint_model_path = "models\\best_keypoints.pt"

# Videó betöltése
frames, fps, width, height = load_video(video_path)

# Modell betöltése
tracker = Tracker(model_path=model_path, video_fps=fps)

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
            "player_coordinates" in stub_data and
            "ball_coordinates" in stub_data):

            tracks = stub_data["tracks"]
            tracker.goalkeeper_ids = stub_data.get("goalkeeper_ids", set())
            keypoint_data = {
                "keypoints": stub_data["keypoints"],
                "player_coordinates": stub_data["player_coordinates"],
                "ball_coordinates": stub_data["ball_coordinates"]
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
    keypoint_data = process_keypoint_annotations(
        video_path,
        keypoint_model_path,
        players_tracks=tracks["players"],
        ball_tracks=tracks["ball"]
    )
    print("Kulcspontok detektálva, játékoskoordináták és labdakoordináták kiszámítva!")

    # Mentés stub fájlba
    stub_data = {
        "tracks": tracks,
        "goalkeeper_ids": tracker.goalkeeper_ids,
        "keypoints": keypoint_data.get("keypoints", []),
        "player_coordinates": keypoint_data.get("player_coordinates", []),
        "ball_coordinates": keypoint_data.get("ball_coordinates", [])
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
    player_coordinates_list=keypoint_data.get("player_coordinates", []),
    ball_coordinates_list=keypoint_data.get("ball_coordinates", [])
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

# Legközelebbi játékosok szűrése és annotálása
closest_player_ids_filtered = closest_player_ids_filter(tracker.closest_player_ids)
annotated_frames = tracker.draw_closest_players_triangles(annotated_frames, closest_player_ids_filtered, tracks)
print("A labdához legközelebbi játékosok szűrése és annotálása befejeződött!")

# Labdabirtoklás számítása és megjelenítése
possession = BallPossession()
annotated_frames = possession.measure_and_draw_possession(annotated_frames, closest_player_ids_filtered)
print("Labdabirtoklás számítása és megjelenítése befejeződött!")

# Passzok számítása és annotálása
pass_counter = PassCounter()
pass_counter.process_passes_per_frame(closest_player_ids_filtered, total_frames=len(frames))
annotated_frames = pass_counter.draw_pass_statistics(annotated_frames)
print("Passzok számítása és annotálása befejeződött!")

# Lesen álló játékosok detektálása és zászló megjelenítése
offside_detector = OffsideDetector(
    player_coordinates=keypoint_data["player_coordinates"],
    ball_coordinates=keypoint_data["ball_coordinates"],
    closest_player_ids_filtered=closest_player_ids_filtered,
    track_id_to_team=tracker.track_id_to_team,
    field_sides=field_sides,
    flag_path="offside_detection/offside_flag.png"
)
offsides_per_frame = offside_detector.detect_offsides_per_frame()
annotated_frames = offside_detector.draw_offside_flags(annotated_frames, offsides_per_frame, tracks)
offside_detector.plot_top5_offsides(
    fps=fps,
    team1_color_rgb=tuple(np.array(tracker.team1_color) / 255.0),
    team2_color_rgb=tuple(np.array(tracker.team2_color) / 255.0)
)
print("Lesen álló játékosok detektálása és megjelenítése befejeződött!")

# Színes négyzetek annotálása
annotated_frames = tracker.coloured_squares_annotations(annotated_frames)
print("Csapatok színével ellátott négyzetek annotálása befejeződött!")

# Játékos pozíciók térfélenkénti grafikonjának generálása
plot_players_per_half_graph(
    player_coordinates_list=keypoint_data["player_coordinates"],
    track_id_to_team=tracker.track_id_to_team,
    field_sides=field_sides,
    team1_color=tracker.team1_color,
    team2_color=tracker.team2_color,
    fps=fps
)
print("Játékos pozíciók térfélenkénti grafikonja elmentve a player_positions_graph mappába!")

# Játékos aktivitási statisztikák generálása
generate_player_activity_summary(
    player_coordinates_list=keypoint_data["player_coordinates"],
    speed_estimator=tracker.speed_estimator,
    tracker=tracker
)
print("Játékos aktivitási statisztikák generálása befejeződött!")

# Hőtérképek generálása
generate_player_heatmaps(keypoint_data.get("player_coordinates", []))
generate_ball_heatmap(keypoint_data.get("ball_coordinates", []))
print("Hőtérképek elmentve a heatmaps mappába!")

# Output videó generálása
generate_output_video(annotated_frames, output_video_path, fps, width, height)
print("Kimeneti videó mentve:", output_video_path)

#print("Kapus ID-k:", tracker.goalkeeper_ids)

# Legközelebbi játékosok szótár kiírása
"""
print("Legközelebbi játékosok szótár:")
for frame_num, (player_id, team_id) in closest_player_ids_filtered.items():
    print(f"Frame {frame_num}: Játékos ID: {player_id} Team ID: {team_id}")
"""
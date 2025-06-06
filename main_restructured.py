import os
import pickle
import numpy as np
from utils import load_video, generate_output_video, get_majority_team_sides, closest_player_ids_filter, save_video_thumbnail
from tracker import Tracker
from pitch_config import process_keypoint_annotations
from heatmaps import generate_player_heatmaps, generate_ball_heatmap
from ball_possession import BallPossession
from passing_measurement import PassCounter
from player_positions_per_frame import plot_players_per_half_graph
from player_activity import generate_player_activity_summary
from offside_detection import OffsideDetector

try:
    from main_nicegui import gui_log
except ImportError:
    def gui_log(msg): print(msg)

def run_analysis_pipeline(video_path: str):
    # --- Elnevezések ---
    filename = os.path.basename(video_path)
    video_stem = os.path.splitext(filename)[0]
    output_video_dir = f"output_videos/annotated_{video_stem}"
    output_video_path = f"{output_video_dir}/annotated_{video_stem}.mp4"
    stub_path = f"stubs/{video_stem}.pkl"
    model_path = "models/best.pt"
    keypoint_model_path = "models/best_keypoints.pt"

    os.makedirs(output_video_dir, exist_ok=True)

    gui_log("\n[1] Videó betöltése...")
    frames, fps, width, height = load_video(video_path)
    tracker = Tracker(model_path=model_path, video_fps=fps)

    # --- Stub betöltése vagy új számítás ---
    tracks = None
    keypoint_data = None
    if os.path.exists(stub_path):
        try:
            with open(stub_path, "rb") as f:
                stub_data = pickle.load(f)
            if all(k in stub_data for k in ("tracks", "goalkeeper_ids", "keypoints", "player_coordinates", "ball_coordinates")):
                tracks = stub_data["tracks"]
                tracker.goalkeeper_ids = stub_data.get("goalkeeper_ids", set())
                keypoint_data = {
                    "keypoints": stub_data["keypoints"],
                    "player_coordinates": stub_data["player_coordinates"],
                    "ball_coordinates": stub_data["ball_coordinates"]
                }
                gui_log("[2] Stub fájl betöltve.")
        except:
            gui_log("[2] Hiba a stub betöltésekor!")

    if tracks is None or keypoint_data is None:
        gui_log("[3] Tracker detektálás és keypoint annotáció indul...")
        tracks = tracker.detect_video(frames, read_from_stub=False, stub_path=None)
        keypoint_data = process_keypoint_annotations(video_path, keypoint_model_path, tracks["players"], tracks["ball"])
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        with open(stub_path, "wb") as f:
            pickle.dump({
                "tracks": tracks,
                "goalkeeper_ids": tracker.goalkeeper_ids,
                "keypoints": keypoint_data["keypoints"],
                "player_coordinates": keypoint_data["player_coordinates"],
                "ball_coordinates": keypoint_data["ball_coordinates"]
            }, f)
        gui_log("[4] Detektálás és annotáció elvégezve, stub mentve.")

    gui_log("[5] Annotált frame-ek készítése...")
    annotated_frames = tracker.annotations(
        frames, tracks,
        keypoints_list=keypoint_data["keypoints"],
        player_coordinates_list=keypoint_data["player_coordinates"],
        ball_coordinates_list=keypoint_data["ball_coordinates"]
    )

    gui_log("[6] Térfél meghatározás...")
    field_sides = get_majority_team_sides(
        player_coordinates=keypoint_data["player_coordinates"],
        players_tracks=tracks["players"],
        first_frame=frames[0],
        team1_color=tracker.team1_color,
        team2_color=tracker.team2_color,
        team_assigner=tracker.teamAssigner
    )

    gui_log("[7] Kapus annotáció és közeli játékosok számítása...")
    annotated_frames = tracker.goalkeeper_annotations(annotated_frames, tracks, frames, keypoint_data["player_coordinates"], field_sides)
    closest_player_ids_filtered = closest_player_ids_filter(tracker.closest_player_ids)
    annotated_frames = tracker.draw_closest_players_triangles(annotated_frames, closest_player_ids_filtered, tracks)

    gui_log("[8] Labdabirtoklás és passz statisztikák számítása...")
    possession = BallPossession()
    annotated_frames = possession.measure_and_draw_possession(annotated_frames, closest_player_ids_filtered)

    pass_counter = PassCounter()
    pass_counter.process_passes_per_frame(closest_player_ids_filtered, len(frames))
    annotated_frames = pass_counter.draw_pass_statistics(annotated_frames)

    gui_log("[9] Lesek detektálása és zászlók kirajzolása...")
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

    gui_log("[10] Egyéb grafikai annotációk...")
    annotated_frames = tracker.coloured_squares_annotations(annotated_frames)

    plot_players_per_half_graph(
        player_coordinates_list=keypoint_data["player_coordinates"],
        track_id_to_team=tracker.track_id_to_team,
        field_sides=field_sides,
        team1_color=tracker.team1_color,
        team2_color=tracker.team2_color,
        fps=fps
    )

    generate_player_activity_summary(
        player_coordinates_list=keypoint_data["player_coordinates"],
        speed_estimator=tracker.speed_estimator,
        tracker=tracker
    )

    generate_player_heatmaps(keypoint_data["player_coordinates"])
    generate_ball_heatmap(keypoint_data["ball_coordinates"])

    gui_log("[11] Videó mentése folyamatban...")
    generate_output_video(annotated_frames, output_video_path, fps, width, height)
    gui_log(f"[11] Kimeneti videó mentve: {output_video_path}")

    gui_log("[12] Thumbnail generálása...")
    save_video_thumbnail(video_path, output_video_path)
    gui_log("[12] Thumbnail mentve.")

    return output_video_path
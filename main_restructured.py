import os
import pickle
import numpy as np
from utils import load_video, generate_output_video, get_majority_team_sides, closest_player_ids_filter, save_video_thumbnail, get_players_with_minimum_presence, get_valid_player_colors, save_all_jersey_images
from tracker import Tracker
from pitch_config import process_keypoint_annotations
from heatmaps import generate_player_heatmaps, generate_ball_heatmap
from ball_possession import BallPossession
from passing_measurement import PassCounter
from player_positions_per_frame import plot_players_per_half_graph
from player_activity import generate_player_activity_summary
from offside_detection import OffsideDetector

# Fő elemzési pipeline
def run_analysis_pipeline(video_path: str, status_callback=None, config=None):
    if config is None:
        config = {}
    # Inicializálás
    filename = os.path.basename(video_path)
    video_stem = os.path.splitext(filename)[0]
    output_video_dir = f"output_videos/annotated_{video_stem}"
    output_video_path = f"{output_video_dir}/annotated_{video_stem}.mp4"
    stub_path = f"stubs/{video_stem}.pkl"
    heatmap_dir = os.path.join(output_video_dir, "heatmaps")
    model_path = "models/best.pt"
    keypoint_model_path = "models/best_keypoints.pt"
    os.makedirs(output_video_dir, exist_ok=True)

    # Callback függvény a státusz üzenetekhez
    def update(msg):
        if status_callback:
            status_callback(msg)

    # Videó betöltése
    frames, fps, width, height = load_video(video_path)
    tracker = Tracker(model_path=model_path, video_fps=fps)

    # Stub betöltése vagy új generálása
    update("Stub fájl betöltése...")
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
                update("Stub fájl sikeresen betöltve!")
        except:
            update("Hiba történt a stub fájl betöltésekor, újraszámítás szükséges.")

    # Új feldolgozás, ha nincs stub
    update("Stub fájl létrehozása...")
    if tracks is None or keypoint_data is None:
        # Objektumdetektálás
        tracks = tracker.detect_video(frames, 
                                      read_from_stub=False, 
                                      stub_path=None, 
                                      status_callback=update)
        # Kulcspontok és koordináták kiszámítása
        keypoint_data = process_keypoint_annotations(video_path, 
                                                     keypoint_model_path, 
                                                     tracks["players"], 
                                                     tracks["ball"], 
                                                     status_callback=update)
        # Stub mentése
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        with open(stub_path, "wb") as f:
            pickle.dump({
                "tracks": tracks,
                "goalkeeper_ids": tracker.goalkeeper_ids,
                "keypoints": keypoint_data["keypoints"],
                "player_coordinates": keypoint_data["player_coordinates"],
                "ball_coordinates": keypoint_data["ball_coordinates"]
            }, f)

    # Annotálás
    update("Annotálás folyamatban...")
    annotated_frames = tracker.annotations(
            frames, 
            tracks,
            keypoints_list=keypoint_data["keypoints"],
            player_coordinates_list=keypoint_data["player_coordinates"],
            ball_coordinates_list=keypoint_data["ball_coordinates"],
            draw_player_ellipses = config.get("show_player_ellipses", True),
            draw_player_ids = config.get("show_player_ids", True),
            draw_referee_ellipses = config.get("show_referees", True),
            draw_ball_triangle = config.get("show_ball_triangle", True),
            draw_speed_distance = config.get("show_speed_distance", True),
            draw_keypoints = config.get("show_keypoints", True),
            draw_player_coordinates = config.get("show_player_coordinates", True)
    )

    # Térfél meghatározása
    update("Térfelek meghatározása...")
    field_sides = get_majority_team_sides(
        player_coordinates=keypoint_data["player_coordinates"],
        players_tracks=tracks["players"],
        first_frame=frames[0],
        team1_color=tracker.team1_color,
        team2_color=tracker.team2_color,
        team_assigner=tracker.teamAssigner
    )

    # Kapusok annotálása
    update("Kapusok annotációja...")
    annotated_frames = tracker.goalkeeper_annotations(annotated_frames, 
                                                      tracks, frames, 
                                                      keypoint_data["player_coordinates"], 
                                                      field_sides,
                                                      draw_goalkeeper_ellipses = config.get("show_player_ellipses", True),
                                                      draw_goalkeeper_ids = config.get("show_player_ids", True)
                                                      )
    closest_player_ids_filtered = closest_player_ids_filter(tracker.closest_player_ids)

    # Labdát birtokló játékosok annotálása
    update("Labdát birtokló játékosok annotálása...")
    draw_closest_player_triangle = config.get("show_closest_player_triangle", True)
    if draw_closest_player_triangle:
        annotated_frames = tracker.draw_closest_players_triangles(annotated_frames, closest_player_ids_filtered, tracks)

    # Labdabirtoklás kiszámítása
    update("Labdabirtoklás számítása és annotálása...")
    possession = BallPossession()
    annotated_frames = possession.measure_and_draw_possession(annotated_frames, 
                                                              closest_player_ids_filtered,
                                                              draw_possession_overlay = config.get("show_possession_overlay", True)
                                                              )

    # Passzok számlálása
    update("Passzok számítása és annotálása...")
    pass_counter = PassCounter()
    pass_counter.process_passes_per_frame(closest_player_ids_filtered, len(frames))
    draw_pass_statistics = config.get("show_pass_statistics")
    if draw_pass_statistics:
        annotated_frames = pass_counter.draw_pass_statistics(annotated_frames)

    # Lesen álló játékosok detektálása
    update("Lesek detektálása és annotálása...")
    offside_detector = OffsideDetector(
        player_coordinates=keypoint_data["player_coordinates"],
        ball_coordinates=keypoint_data["ball_coordinates"],
        closest_player_ids_filtered=closest_player_ids_filtered,
        track_id_to_team=tracker.track_id_to_team,
        field_sides=field_sides,
        flag_path="offside_detection/offside_flag.png"
    )
    offsides_per_frame = offside_detector.detect_offsides_per_frame()
    draw_offside_flags = config.get("show_offside_flags", True)
    if draw_offside_flags:
        annotated_frames = offside_detector.draw_offside_flags(annotated_frames, offsides_per_frame, tracks)
    statistics_dir = os.path.join(output_video_dir, "statistics")
    draw_offside_stats = config.get("show_offside_stats", True)
    if draw_offside_stats:
        offside_detector.plot_top5_offsides(
            fps=fps,
            output_dir=statistics_dir,
            team1_color_rgb=tuple(np.array(tracker.team1_color) / 255.0),
            team2_color_rgb=tuple(np.array(tracker.team2_color) / 255.0)
        )

    # Csapat színek megjelenítése
    update("Csapatok jelölése és színezése...")
    annotated_frames = tracker.coloured_squares_annotations(annotated_frames, 
                                                            draw_team_colors_topbar = config.get("show_team_colors_topbar", True)
                                                            )

    # Játékosok száma térfelenként a megadott időintervallumokban grafikon elkészítése
    update("Grafikonok és statisztikák készítése...")
    statistics_dir = os.path.join(output_video_dir, "statistics")
    draw_players_per_half_graph = config.get("show_players_per_half_graph", True)
    if draw_players_per_half_graph:
        plot_players_per_half_graph(
            player_coordinates_list=keypoint_data["player_coordinates"],
            track_id_to_team=tracker.track_id_to_team,
            field_sides=field_sides,
            team1_color=tracker.team1_color,
            team2_color=tracker.team2_color,
            fps=fps,
            output_dir=statistics_dir
        )

    # Játékos aktivitási statisztikák generálása
    statistics_dir = os.path.join(output_video_dir, "statistics")
    draw_player_activity_stats = config.get("show_player_activity_stats", True)
    if draw_player_activity_stats:
        generate_player_activity_summary(
            player_coordinates_list=keypoint_data["player_coordinates"],
            speed_estimator=tracker.speed_estimator,
            tracker=tracker,
            output_dir=statistics_dir
        )

    # Játékosok és labda hőtérképek generálása
    draw_player_heatmaps = config.get("show_player_heatmaps", True)
    if draw_player_heatmaps:
        generate_player_heatmaps(keypoint_data["player_coordinates"], 
                                track_id_to_team=tracker.track_id_to_team, 
                                output_dir = heatmap_dir,
                                tracker=tracker)
    
    ball_dir = os.path.join(heatmap_dir, "ball_heatmap")
    draw_ball_heatmap = config.get("show_ball_heatmap")
    if draw_ball_heatmap:
        generate_ball_heatmap(keypoint_data["ball_coordinates"], 
                            output_path=os.path.join(ball_dir, "ball_heatmap.png"))
        
    # Játékosok egyéni statisztikáinak elkészítése
    update("Színes mezkártyák rajzolása és mentése...")
    valid_players = get_players_with_minimum_presence(tracks["players"], minimum_ratio=0.5)
    valid_player_colors = get_valid_player_colors(
        valid_players,
        tracker.track_id_to_team,
        tracker.team1_color,
        tracker.team2_color
    )
    save_all_jersey_images(
        valid_player_colors=valid_player_colors,
        track_id_to_team=tracker.track_id_to_team,
        video_name=video_stem
    )

    # Kimeneti videó mentése + előnézeti kép generálása
    update("Kimeneti videó generálása és thumbnail készítése...")
    generate_output_video(annotated_frames, output_video_path, fps, width, height)
    save_video_thumbnail(video_path, output_video_path)

    return output_video_path
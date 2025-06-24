import os
import json
from utils import get_players_with_minimum_presence, get_players_presence_ratios

def generate_basic_player_statistics(players_tracks, 
                                     track_id_to_team, 
                                     speed_estimator,
                                     individual_passes,
                                     ball_possession_count,
                                     output_dir: str,
                                     offside_frame_counts,
                                     fps,
                                     minimum_ratio: float = 0.5):
    
    # Megfelelő játékosok kiválasztása
    valid_players = get_players_with_minimum_presence(players_tracks, minimum_ratio)

    # Jelenléti arány a frameken
    presence_ratios = get_players_presence_ratios(players_tracks)

    player_stats = {}
    for track_id in valid_players:
        track_id_str = str(track_id)

        # Alap értékek
        team = int(track_id_to_team.get(track_id, -1))
        presence_ratio = round(presence_ratios.get(track_id, 0.0), 4)
        distance_m = float(round(speed_estimator.get_player_distance_m(track_id), 2))

        # Sebességek
        player_info = speed_estimator.player_data.get(track_id, {})
        smoothed_speeds = player_info.get("smoothed_list", [])

        if smoothed_speeds:
            avg_speed_kmh = float(round((sum(smoothed_speeds) / len(smoothed_speeds)) * 3.6, 2))
            max_speed_kmh = float(round(max(smoothed_speeds) * 3.6, 2))
        else:
            avg_speed_kmh = 0.0
            max_speed_kmh = 0.0

        # Passzok
        accurate_passes = int(individual_passes.get(track_id, {}).get("accurate", 0))
        inaccurate_passes = int(individual_passes.get(track_id, {}).get("inaccurate", 0))

        # Labdabirtoklás
        ball_possession = int(ball_possession_count.get(track_id, 0))

        # Lesen töltött idő
        offside_frames = offside_frame_counts.get(track_id, 0)
        total_seconds = offside_frames / fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds - int(total_seconds)) * 1000)
        offside_time_str = f"{minutes:02d}:{seconds:02d}:{milliseconds:03d}"

        # JSON szerkezet feltöltése
        player_stats[track_id_str] = {
            "track_id": int(track_id),
            "team": team,
            "presence_ratio": presence_ratio,
            "distance_m": float(f"{distance_m:.2f}"),
            "avg_speed_kmh": avg_speed_kmh,
            "max_speed_kmh": max_speed_kmh,
            "accurate_passes": accurate_passes,
            "inaccurate_passes": inaccurate_passes,
            "ball_possession_count": ball_possession,
            "offside_time": offside_time_str
        }

    # Mentés
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "player_stats.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(player_stats, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Játékos statisztikák elmentve: {output_path}")

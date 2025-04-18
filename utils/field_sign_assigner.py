from collections import defaultdict, Counter
import numpy as np
from utils.team_assigner_utils import TeamAssigner

# Meghatározza az adott frame-en a csapatok térfél-hozzárendelését a játékosok koordinátái alapján
def determine_team_sides(player_coordinates_frame: dict[int, tuple[float, float]], player_to_team: dict[int, int]) -> dict[int, str] | None:
    team_positions = defaultdict(list)

    for track_id, (x, _) in player_coordinates_frame.items():
        team = player_to_team.get(track_id)
        if team:
            team_positions[team].append(x)

    if len(team_positions) < 2:
        return None  # nem tudunk dönteni, ha csak 1 csapat van jelen

    team_averages = {team: np.mean(xs) for team, xs in team_positions.items()}

    if team_averages[1] < team_averages[2]:
        return {1: 'left', 2: 'right'}
    else:
        return {1: 'right', 2: 'left'}
    
def get_majority_team_sides(player_coordinates: list[dict], players_tracks: list[dict], first_frame, team1_color, team2_color, team_assigner: TeamAssigner) -> dict[int, str]:

    # Játékosok csapatának meghatározása
    player_to_team = {}

    for frame in players_tracks:
        for track_id, player in frame.items():
            if track_id in player_to_team:
                continue
            upper_body = team_assigner.get_upper_body_image(first_frame, player["bbox"])
            color = team_assigner.get_player_color(upper_body)
            team_id = team_assigner.get_player_to_team(color, team1_color, team2_color)
            player_to_team[track_id] = team_id

    # Minden frame-re meghatározzuk a csapatok térfél-hozzárendelését
    votes = []

    for frame_coords in player_coordinates:
        side_result = determine_team_sides(frame_coords, player_to_team)
        if side_result:
            votes.append(tuple(sorted(side_result.items())))

    if not votes:
        return {}

    # Leggyakoribb térfél-hozzárendelés meghatározása
    most_common = Counter(votes).most_common(1)[0][0]
    return dict(most_common)

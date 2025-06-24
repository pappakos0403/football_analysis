from collections import defaultdict

def closest_player_ids_filter(closest_player_ids_filtered):
    # Eredmény tárolása
    filtered_closest = {}

    # Ideiglenes változók a streak nyomon követésére
    current_id = None
    current_team = None
    streak_count = 0
    streak_frames = []

    # Végigmegyünk az összes frame-en időrendben
    for frame_num, (player_id, team_id) in closest_player_ids_filtered.items():
        if player_id == current_id and player_id is not None:
            # Ha ugyanaz az ID, növeljük a streaket
            streak_count += 1
            streak_frames.append(frame_num)
        else:
            # Ha a streak véget ér, ellenőrizzük a hosszát
            if streak_count >= 3:
                # Csak akkor hagyjuk meg, ha legalább elérte a küszöböt
                for f in streak_frames:
                    filtered_closest[f] = (current_id, current_team)
            else:
                # Ha kevesebb volt, akkor None-t írunk be
                for f in streak_frames:
                    filtered_closest[f] = (None, None)

            # Új streak indítása
            current_id = player_id
            current_team = team_id
            streak_count = 1
            streak_frames = [frame_num]

    # Utolsó streak ellenőrzése a végén
    if streak_count >= 2:
        for f in streak_frames:
            filtered_closest[f] = (current_id, current_team)
    else:
        for f in streak_frames:
            filtered_closest[f] = (None, None)

    # Ha kimaradtak frame-ek, azokat pótoljuk (None, None)-nal
    for frame_num in closest_player_ids_filtered.keys():
        if frame_num not in filtered_closest:
            filtered_closest[frame_num] = (None, None)

    return filtered_closest

# Megszámolja, hogy egy játékos hányszor volt labdabirtokló
def count_ball_possessions_per_player(closest_player_ids_filtered: dict) -> dict:

    possession_counter = defaultdict(int)
    previous_player_id = None

    for frame_id in sorted(closest_player_ids_filtered.keys()):
        player_id, _ = closest_player_ids_filtered[frame_id]

        if player_id is None:
            previous_player_id = None
            continue

        if player_id != previous_player_id:
            possession_counter[player_id] += 1

        previous_player_id = player_id

    return dict(possession_counter)
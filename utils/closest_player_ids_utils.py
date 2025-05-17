def closest_player_ids_filter(closest_player_ids):
    # Eredmény tárolása
    filtered_closest = {}

    # Ideiglenes változók a streak nyomon követésére
    current_id = None
    current_team = None
    streak_count = 0
    streak_frames = []

    for frame_num, (player_id, team_id) in closest_player_ids.items():
        if player_id == current_id:
            # Ha folytatódik az előző streak
            streak_count += 1
            streak_frames.append(frame_num)
        else:
            # Ha az előző streak véget ér, ellenőrizzük a hosszát
            if streak_count >= 3:
                # Csak akkor hagyjuk meg, ha legalább 3 volt
                for f in streak_frames:
                    filtered_closest[f] = (current_id, current_team)
            else:
                # Ha kevesebb volt, akkor mind None
                for f in streak_frames:
                    filtered_closest[f] = (None, None)

            # Új streak indítása
            current_id = player_id
            current_team = team_id
            streak_count = 1
            streak_frames = [frame_num]

    # Utolsó streak ellenőrzése a végén
    if streak_count >= 3:
        for f in streak_frames:
            filtered_closest[f] = (current_id, current_team)
    else:
        for f in streak_frames:
            filtered_closest[f] = (None, None)

    # Az üres frame-eket is pótoljuk
    for frame_num in closest_player_ids.keys():
        if frame_num not in filtered_closest:
            filtered_closest[frame_num] = (None, None)

    return filtered_closest
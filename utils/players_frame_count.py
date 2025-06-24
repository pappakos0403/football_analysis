from collections import defaultdict

# Listát ad azokról a játékosokról, akik legalább a frame-k 50%-án detektálva vannak
def get_players_with_minimum_presence(players_tracks: list, minimum_ratio: float = 0.5) -> list:

    # Jelenlétszámláló minden track_id-re
    presence_counter = defaultdict(int)
    total_frames = len(players_tracks)

    # Végigmegyünk minden frame-en
    for frame_data in players_tracks:
        for track_id in frame_data.keys():
            presence_counter[track_id] += 1

    # Szűrés: csak azok, akik legalább a minimum_ratio * összes képkocka arányban jelen voltak
    valid_players = [
        track_id for track_id, count in presence_counter.items()
        if count >= total_frames * minimum_ratio
    ]

    return valid_players

# Játékosok a framek hány %-án voltak detektálva
def get_players_presence_ratios(players_tracks: list) -> dict:

    presence_counter = defaultdict(int)
    total_frames = len(players_tracks)

    for frame_data in players_tracks:
        for track_id in frame_data.keys():
            presence_counter[track_id] += 1

    presence_ratios = {
        track_id: count / total_frames
        for track_id, count in presence_counter.items()
    }

    return presence_ratios
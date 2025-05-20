# passing_measurement.py

import cv2

class PassCounter:
    def __init__(self):
        # Passz számlálók listája minden frame-re
        self.stats_per_frame = []

        # Előző birtokos adatai
        self.prev_player_id = None
        self.prev_team_id = None

        # Folyamatos számlálók
        self.team1_accurate = 0
        self.team1_inaccurate = 0
        self.team2_accurate = 0
        self.team2_inaccurate = 0

    def process_passes_per_frame(self, closest_player_ids_filtered: dict, total_frames: int):
        """
        Frame-enkénti feldolgozás, eltároljuk minden frame-re az akkori statisztikát.
        """
        for frame_num in range(total_frames):
            player_id, team_id = closest_player_ids_filtered.get(frame_num, (None, None))

            if player_id is not None and team_id is not None:
                if self.prev_player_id is not None and self.prev_player_id != player_id:
                    # Új birtokos → passz történt
                    if self.prev_team_id == team_id:
                        # Pontos passz
                        if team_id == 1:
                            self.team1_accurate += 1
                        else:
                            self.team2_accurate += 1
                    else:
                        # Pontatlan passz
                        if self.prev_team_id == 1:
                            self.team1_inaccurate += 1
                        elif self.prev_team_id == 2:
                            self.team2_inaccurate += 1

                # Frissítjük az előző játékost
                self.prev_player_id = player_id
                self.prev_team_id = team_id

            # Minden frame-hez elmentjük az akkori statisztikát
            self.stats_per_frame.append((
                self.team1_accurate,
                self.team1_inaccurate,
                self.team2_accurate,
                self.team2_inaccurate
            ))

    def draw_pass_statistics(self, frames):
        """
        Frame-enként megjeleníti a passz statisztikát a bal felső sarokban.
        """
        for i, frame in enumerate(frames):
            if i >= len(self.stats_per_frame):
                break  # ne lépjünk túl

            team1_acc, team1_inacc, team2_acc, team2_inacc = self.stats_per_frame[i]

            overlay = frame.copy()
            # Átlátszó téglalap rajzolása a bal felső sarokba
            cv2.rectangle(overlay, (0, 0), (640, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Szöveg kirajzolása
            cv2.putText(frame, f"Team1: Pontos passzok: {team1_acc} Pontatlan passzok: {team1_inacc}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            cv2.putText(frame, f"Team2: Pontos passzok: {team2_acc} Pontatlan passzok: {team2_inacc}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return frames

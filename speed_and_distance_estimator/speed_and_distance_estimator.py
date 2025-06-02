import numpy as np

class SpeedAndDistanceEstimator:
    def __init__(self, fps, smoothing_window_size=8, max_realistic_speed_kmh=36):
        self.fps = fps
        if self.fps <= 0:
            raise ValueError("Az FPS értékének pozitívnak kell lennie.")
        # Mozgóátlag ablakméret beállítása
        self.smoothing_window_size = max(1, smoothing_window_size)
        # Maximális reális sebesség m/s-ban, átszámítva km/h-ból
        self.max_realistic_speed_ms = max_realistic_speed_kmh * 1000 / 3600  
        # Játékosok adatainak tárolása: track_id -> pozíciók, időbélyegek, teljes megtett távolság, sebességek, simított sebesség
        self.player_data = {}

    def add_measurement(self, track_id, position_m, frame_num, team_id=None):
        # Időbélyeg kiszámítása a képkocka sorszám és az fps alapján
        timestamp = frame_num / self.fps

        # Ha még nem létezik az adott játékos az adatbázisban, inicializáljuk
        if track_id not in self.player_data:
            self.player_data[track_id] = {
                'positions': [],
                'timestamps': [],
                'total_distance': 0.0,
                'speeds': [],
                'smoothed_speed': 0.0,
                'team_id': team_id
            }

        # Az aktuális játékos adatai
        player_info = self.player_data[track_id]
        instant_speed_ms = 0.0

        # Ha már vannak korábbi pozíciók, akkor kiszámítjuk a távolságot és a sebességet
        if player_info['positions']:
            prev_pos = np.array(player_info['positions'][-1])
            current_pos = np.array(position_m)
            prev_time = player_info['timestamps'][-1]

            # Két pozíció közötti távolság
            delta_distance = np.linalg.norm(current_pos - prev_pos)
            delta_time = timestamp - prev_time

            if delta_time > 0:
                # Sebesség kiszámítása: távolság/idő
                calculated_speed_ms = delta_distance / delta_time

                # Sebesség korlátozása a maximális reális értékre
                if calculated_speed_ms > self.max_realistic_speed_ms:
                    instant_speed_ms = self.max_realistic_speed_ms
                elif calculated_speed_ms < 0:
                    instant_speed_ms = 0.0
                else:
                    instant_speed_ms = calculated_speed_ms
                
                # Megtett távolság hozzáadása az összesített értékhez
                player_info['total_distance'] += delta_distance
            else:
                # Ha nincs időeltolódás, a sebesség az utolsó mért érték marad
                instant_speed_ms = player_info['speeds'][-1] if player_info['speeds'] else 0.0
        else:
            # Ha ez az első pozíció, a sebesség nulla
            instant_speed_ms = 0.0

        # Új pozíció és időbélyeg mentése
        player_info['positions'].append(list(position_m))
        player_info['timestamps'].append(timestamp)
        player_info['speeds'].append(instant_speed_ms)

        # Sebesség simítása mozgóátlaggal
        if len(player_info['speeds']) >= self.smoothing_window_size:
            window = player_info['speeds'][-self.smoothing_window_size:]
            player_info['smoothed_speed'] = np.mean(window)
        elif player_info['speeds']:
            player_info['smoothed_speed'] = np.mean(player_info['speeds'])

        # Simított sebességek gyűjtése a listába
        if 'smoothed_list' not in player_info:
            player_info['smoothed_list'] = []
        player_info['smoothed_list'].append(player_info['smoothed_speed'])

    def get_player_speed_kmh(self, track_id):
        # A játékos aktuális simított sebessége km/h-ban visszaadva
        if track_id in self.player_data:
            return self.player_data[track_id]['smoothed_speed'] * 3.6
        return 0.0

    def get_player_distance_m(self, track_id):
        # A játékos által megtett teljes távolság méterben visszaadva
        if track_id in self.player_data:
            return self.player_data[track_id]['total_distance']
        return 0.0

    def get_player_info(self, track_id):
        # A játékos sebességének és megtett távolságának lekérdezése
        speed_kmh = self.get_player_speed_kmh(track_id)
        distance_m = self.get_player_distance_m(track_id)
        return speed_kmh, distance_m

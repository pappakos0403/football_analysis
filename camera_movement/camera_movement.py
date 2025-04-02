import cv2
import numpy as np
import pickle
import os

class CameraMovement:
    def __init__(self, first_frame):
        # Minimum távolság, amely felett elmozdulást detektálunk
        self.minimum_distance = 5

        # Optikai áramlás paraméterek
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Első képkocka szürkeárnyalatos képe
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # Maszk inicializálása a jellemzőpontokhoz (kép szélein)
        mask = np.zeros_like(first_gray)
        mask[:, 0:20] = 1  # bal szél
        mask[:, 900:1050] = 1  # jobb szél (felbontástól függően testreszabható)

        # Jellemzőpontok detektálásához szükséges paraméterek
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask
        )

     # Kamera elmozdulás kiszámítása az összes frame alapján
    def calculate_movement(self, frames, read_from_stub=False, stub_path=None):
        # Stub fájlból beolvasás
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Mozgás inicializálása (kezdetben minden frame mozgása 0,0)
        camera_movement = [[0, 0]] * len(frames)

        # Kezdő frame szürkeárnyalatos képe és jellemzőpontjai
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.feature_params)

        # Végigmegyünk a frame-eken
        for i in range(1, len(frames)):
            # Aktuális frame szürkeárnyalatos képe
            frame_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            # Új jellemzőpontok meghatározása optikai áramlással
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            # Elmozdulás meghatározása (legnagyobb vektor alapján)
            max_distance = 0
            dx, dy = 0, 0

            for new, old in zip(new_features, old_features):
                # Koordináták kinyerése
                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()

                # Euklideszi távolság a két pont között
                distance = np.linalg.norm([x_new - x_old, y_new - y_old])

                # Ha ez a legnagyobb mozgás eddig, elmentjük
                if distance > max_distance:
                    max_distance = distance
                    dx = x_old - x_new
                    dy = y_old - y_new

            # Csak akkor mentjük a mozgást, ha a távolság elég nagy
            if max_distance > self.minimum_distance:
                camera_movement[i] = [dx, dy]

                # Új jellemzőpontokat használjuk a következő körben
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.feature_params)

            # Frissítjük az előző frame-et
            old_gray = frame_gray.copy()

        # Ha meg van adva stub útvonal, elmentjük a mozgásokat
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement
    
    # Tracks pozícióinak módosítása kameramozgás alapján
    def adjust_tracks(self, tracks, camera_movement_per_frame):
        # Végigmegyünk minden objektumtípuson (játékos, bíró, labda)
        for object_type, object_tracks in tracks.items():

            for frame_num, track in enumerate(object_tracks):
                # Az adott frame-hez tartozó kameramozgás (dx, dy)
                dx, dy = camera_movement_per_frame[frame_num]

                # Minden objektum pozícióját korrigáljuk
                for track_id, track_info in track.items():
                    # Eredeti pozíció (bounding box)
                    bbox = track_info.get("bbox", None)

                    # Ha nincs bbox (pl. interpolált labdánál előfordulhat), kihagyjuk
                    if bbox is None or len(bbox) != 4:
                        continue

                    # Bounding box középpontjának kiszámítása
                    x1, y1, x2, y2 = bbox
                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)

                    # Kamera elmozdulásával korrigált pozíció
                    x_adj = x_center - dx
                    y_adj = y_center - dy

                    # Új kulcs hozzáadása a track-hez
                    track_info["position_adjusted"] = (x_adj, y_adj)

        # Kameramozgás vizualizálása: szöveg kiírása a képkockákra
    def draw_movement(self, frames, camera_movement_per_frame):
        # Lista az annotált képkockákhoz
        output_frames = []

        # Végigmegyünk minden képkockán
        for i, frame in enumerate(frames):
            # Képkocka másolása (ne írjuk felül az eredetit)
            frame_copy = frame.copy()

            # Fehér átlátszó overlay téglalap az információ háttérhez
            overlay = frame_copy.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)

            # Elmozdulás kiírása (X és Y komponens)
            dx, dy = camera_movement_per_frame[i]
            text1 = f"Camera Movement X: {dx:.2f}"
            text2 = f"Camera Movement Y: {dy:.2f}"

            # Szöveg rajzolása a képre
            cv2.putText(frame_copy, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame_copy, text2, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            # Annotált képkocka mentése a listába
            output_frames.append(frame_copy)

        # Annotált képkockák listájának visszaadása
        return output_frames

import cv2  # OpenCV a képfeldolgozáshoz
import numpy as np  # numpy a tömbkezeléshez
from ultralytics import YOLO  # YOLO modell a keypoint detektáláshoz
from pitch_config import FootballPitchConfiguration  # Pálya konfiguráció importálása
from view_transformer import ViewTransformer  # Homográfia transzformációhoz szükséges osztály
import os

# Fájl elérési útvonalak
INPUT_VIDEO_PATH = "input_videos//08fd33_4.mp4"      # Bemeneti videó (FULLHD: 1920x1080)
OUTPUT_VIDEO_PATH = "output_videos//test_keypoints.avi"  # Kimeneti videó mentési helye
MODEL_PATH = "models//best_keypoints.pt"               # Keypoint detektáló modell elérési útvonala

# Videó megnyitása, paraméterek lekérése
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)  # Videó megnyitása
if not cap.isOpened():  # Ellenőrizzük a megnyitást
    print("Nem sikerült megnyitni a videót!")
    exit()
fps = cap.get(cv2.CAP_PROP_FPS)  # Képkocka/másodperc
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Videó szélessége (1920)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Videó magassága (1080)
print(f"Input videó felbontása: {frame_width}x{frame_height}")

# Kimeneti videó létrehozása FULLHD felbontásban
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Videó kodek beállítása
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))  # VideoWriter inicializálása

# Modell betöltése (640x640-es képeken tréningezve)
model = YOLO(MODEL_PATH)  # Modell betöltése

# Detektált keypointok skálázási tényezői 640-es -> FULLHD-re
scale_x_det = frame_width / 640.0  # x tengely skála (1920/640)
scale_y_det = frame_height / 640.0  # y tengely skála (1080/640)

# Pálya konfiguráció betöltése a pitch_config.py segítségével
pitch_config = FootballPitchConfiguration()  # Pálya konfiguráció példány

# A pitch_config pontjainak FULLHD koordinátákra történő átskálázása
scale_target_x = frame_width / pitch_config.length  # x skála: 1920 / pálya hossza
scale_target_y = frame_height / pitch_config.width   # y skála: 1080 / pálya szélessége
pitch_vertices = np.array(pitch_config.vertices, dtype=np.float32)  # Pálya vertex-ek numpy tömb
for i in range(len(pitch_vertices)):  # Minden vertex átskálázása FULLHD-re
    x, y = pitch_vertices[i]
    pitch_vertices[i] = [x * scale_target_x, y * scale_target_y]

# Konfidencia küszöb: csak megbízható pontok (confidence ≥ 0.5)
confidence_threshold = 0.5

frame_count = 0  # Képkocka számláló

while True:
    ret, fullhd_frame = cap.read()  # FULLHD képkocka beolvasása
    if not ret:  # Ha nincs több képkocka, kilépünk
        break

    annotated_frame = fullhd_frame.copy()  # Másolat az annotációkhoz

    # Átméretezzük a képet 640x640-re a modell számára
    resized_frame = cv2.resize(fullhd_frame, (640, 640))  # 640x640-es kép

    # Modell futtatása az átméretezett képen
    results = model(resized_frame)
    result = results[0]  # Az első detektált objektum eredménye

    # Ellenőrizzük, hogy elérhető-e keypoint adat
    if hasattr(result, "keypoints") and result.keypoints is not None:
        # Kinyerjük a keypointokat 640x640-es koordinátákban
        kp_array = result.keypoints.xy.cpu().numpy()[0]  # Alak: (N, 2), N általában 32
        conf_array = result.keypoints.conf.cpu().numpy()[0]  # Confidence értékek tömbje (N,)

        # Skálázzuk vissza a detektált pontokat FULLHD koordinátákra
        detected_points = np.zeros_like(kp_array)
        for i, (x, y) in enumerate(kp_array):  # Minden pont átskálázása
            detected_points[i] = [x * scale_x_det, y * scale_y_det]

        # Szűrés: csak a megbízható pontok (confidence ≥ threshold)
        valid_filter = conf_array >= confidence_threshold  
        if np.sum(valid_filter) < 4:  # Ha kevesebb mint 4 referenciapont van, nem számolunk homográfiát
            print(f"Frame {frame_count}: Nincs elegendő referencia pont!")
            out.write(annotated_frame)
            frame_count += 1
            continue

        # Referencia pontok kiválasztása: detektált és a pitch vertex-ek a valid indexek alapján
        frame_reference_points = detected_points[valid_filter]  # Detektált referencia pontok (FULLHD)
        pitch_reference_points = pitch_vertices[valid_filter]  # Pálya referencia pontok (FULLHD)

        # Homográfia számítása a valid referencia pontok alapján
        transformer = ViewTransformer(
            source=pitch_reference_points,   # Forrás: a pálya referencia pontjai
            target=frame_reference_points     # Cél: a detektált referencia pontok
        )

        # A pálya összes pontjának transzformálása a homográfiával (korrigált pontok)
        corrected_points = transformer.transform_points(points=pitch_vertices)

        # Ha egy pont megbízható, akkor a korrigált pont helyét a detektált pontra állítjuk
        for i in range(len(detected_points)):
            if conf_array[i] >= confidence_threshold:
                corrected_points[i] = detected_points[i]

        # Nem rajzoljuk a detektált (rózsaszín) pontokat, csak a korrigált (világoskék) pontokat

        # Rajzoljuk a korrigált pontokat világoskék színnel (például BGR: (230,216,173))
        for (x, y) in corrected_points:
            cv2.circle(annotated_frame, (int(x), int(y)), radius=5, color=(230, 216, 173), thickness=-1)
        
        # Kössük össze a korrigált pontokat sötétkék vonalakkal (BGR: (139, 0, 0)) a pitch_config élei szerint
        for edge in pitch_config.edges:  # Iterálunk az éleken
            i, j = edge  # Az élpárok 1-indexű pontokat tartalmaznak
            pt1 = corrected_points[i - 1]  # Első végpont (0-indexelés)
            pt2 = corrected_points[j - 1]  # Második végpont (0-indexelés)
            cv2.line(annotated_frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (139, 0, 0), 2)
    else:
        print(f"Frame {frame_count}: Nem érhető el keypoint adat!")
    
    out.write(annotated_frame)  # Annotált képkocka írása a kimeneti videóba
    frame_count += 1  # Képkocka számláló növelése

# Erőforrások felszabadítása
cap.release()  # Videó felszabadítása
out.release()  # Kimeneti videó lezárása
print(f"Feldolgozva: {frame_count} képkocka. Kimeneti videó mentve: {OUTPUT_VIDEO_PATH}")

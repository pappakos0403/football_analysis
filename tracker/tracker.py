from ultralytics import YOLO
from utils.video_utils import load_video, generate_output_video
from utils.bbox_utils import get_center_of_bbox, get_bbox_width
from utils.team_assigner_utils import TeamAssigner
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np

def draw_ellipse(frame, bbox, color, track_id = None):
    # Alsó koordináta a bbox alapján
    y2 = int(bbox[3])  

    # Középpont és szélesség kiszámítása a bbox segítségével
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    # Elipszis rajzolása a játékos alá
    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width), int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4
    )

    # Track_id megjelenítése a játékos alatt
    if track_id is not None:
        # Text generálása
        text = str(track_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size

        # Téglalap pozicionálása
        rect_top_left = (x_center - text_w // 2, y2 - text_h - 10)
        rect_bottom_right = (x_center + text_w // 2 + 2 , y2 - 10)

        # Téglalap kitöltése fehér színnel
        cv2.rectangle(frame, rect_top_left, rect_bottom_right, color=(255, 255, 255), thickness=cv2.FILLED)
        cv2.putText(frame, text, (x_center - text_w // 2, y2 - 10), font, font_scale, (0, 0, 0), thickness)
    
    return frame
    
def detect_video(video_path, output_video_path, model):

    # Input videó beolvasása
    frames, fps, width, height = load_video(video_path)

    # Csapatok színéhez szüksges változók
    team1_color = None
    team2_color = None
    threshold = 70
    
    annotated_frames = []

    # DeepSortTracker inicializálása
    deepsort = DeepSort(max_age=30, n_init=10)

    # Perzisztens trackerek számára szükséges változók
    persistens_ids = {}     # fixed_id : apperance_feature (szín jellemző)
    track_to_fixed = {}     # DeepSort aktuális track id -> fixed_id hozzárendelés
    next_fixed_id = 0   # Új fix ID-k generálásához
    apperance_threshold = 30    # Küszöbérték a színjellemzők összehasonlításához

    # Detektálás az összes képkockán
    for frame_num, frame in enumerate(frames):
        annotated_frame = frame.copy()

        # YOLO detekció
        results = model(frame)

        # Az eredményekből a bounding boxok kinyerése
        boxes = results[0].boxes

        # Kapus átállítása játékosra
        # boxes.data tensor oszlopainál a cls átállítása 2-ről 1-re (kapus -> játékos)
        boxes_data = boxes.data.clone()
        boxes_data.data[boxes.data[:, 5] == 1, 5] = 2
        boxes.data = boxes_data

        # boxes.data numpy tömbbé alakítása
        boxes_np = boxes.data.cpu().numpy() if hasattr(boxes.data, "cpu") else boxes.data

        # Külön kezeljük a játékosokat, a játékvezetőket és a labdát
        detected_objects = {
            "players": [], # cls: 2
            "referees": [], # cls: 3
            "ball": [] # cls: 0
        }

        # Tracker számára a detektált bounding boxok listája
        detections_for_tracker = []

        # Bounding boxok feldolgozása
        for box in boxes_np:
            x1, y1, x2, y2, conf, cls = box
            cls = int(cls)

            if conf < 0.7:
                continue

            # Játékosok
            if cls == 2:
                detected_objects["players"].append([x1, y1, x2, y2])
                # bbox = [x1, y1, x2, y2]
                # color = (0, 255, 0)
                # annotated_frame = draw_ellipse(annotated_frame, bbox, color, track_id=player_counter)
                # player_counter += 1
                width_box = x2 - x1
                height_box = y2 - y1
                bbox_list = [float(x1), float(y1), float(width_box), float(height_box)]
                detections_for_tracker.append([bbox_list, float(conf), "player"])

            # Játékvezetők
            elif cls == 3:
                detected_objects["referees"].append([x1, y1, x2, y2])

            # Labda
            elif cls == 0:
                detected_objects["ball"].append([x1, y1, x2, y2])

        tracks = deepsort.update_tracks(detections_for_tracker, frame=frame)

        # Játékos színének meghatározása
        teamAssigner = TeamAssigner()

        if frame_num == 0:

            for id, player in enumerate(detected_objects["players"]):

                # Első csapat színének meghatározása
                if team1_color is None:
                    upper_body_image = teamAssigner.get_upper_body_image(frame, player, id)
                    team1_color = teamAssigner.get_player_color(upper_body_image, id)
                    continue

                # Második csapat színének meghatározása
                upper_body_image = teamAssigner.get_upper_body_image(frame, player, id)
                player_color = teamAssigner.get_player_color(upper_body_image, id)

                # Színkülönbség kiszámítása Euklideszi távolsággal
                color_diff = np.linalg.norm(np.array(team1_color) - np.array(player_color))
                if color_diff > threshold:
                    team2_color = player_color
                    break   
        
        # Elipszis és ID megjelenítése minden követett játékoson
        for track in tracks:
            # Ha a track nincs megerősítve, akkor nem jelenítjük meg
            if not track.is_confirmed():
                continue

            # Bbox kinyerése (formátum: [x1, y1, x2, y2])
            bbox = track.to_tlbr()
            track_id = track.track_id

            # Csapatszín meghatározása a TeamAssigner-rel
            upper_body_image = teamAssigner.get_upper_body_image(frame, bbox, track_id)
            if upper_body_image.size == 0:
                print(f"Hiba: A(z) {track_id}. játékos felső testének képe üres! Klaszterezés kihagyása.")
                continue
            apperance_feature = teamAssigner.get_player_color(upper_body_image, track_id)

            if track_id in track_to_fixed:
                fixed_id = track_to_fixed[track_id]
                persistens_ids[fixed_id] = (persistens_ids[fixed_id] + apperance_feature) / 2.0
            else:
                assigned = False
                for fixed, stored_feature in persistens_ids.items():
                    distance = np.linalg.norm(apperance_feature - stored_feature)
                    if distance < apperance_threshold:
                        fixed_id = fixed
                        track_to_fixed[track_id] = fixed
                        persistens_ids[fixed_id] = (stored_feature + apperance_feature) / 2.0
                        assigned = True
                        break
                if not assigned:
                    fixed_id = next_fixed_id
                    next_fixed_id += 1
                    persistens_ids[fixed_id] = apperance_feature
                    track_to_fixed[track_id] = fixed_id

            """
            player_color = teamAssigner.get_player_color(upper_body_image, track_id)
            if player_color is None:
                print(f"Hiba: A(z) {track_id}. játékos színének meghatározása sikertelen! Klaszterezés kihagyása.")
                continue
            """

            team_color_num = teamAssigner.get_player_to_team(apperance_feature, team1_color, team2_color, track_id)

            # Elipszis rajzolása a megfelelő színnel
            if team_color_num == 1:
                annotated_frame = draw_ellipse(annotated_frame, bbox, team1_color, track_id=track_id)
            elif team_color_num == 2:
                annotated_frame = draw_ellipse(annotated_frame, bbox, team2_color, track_id=track_id)

        # Annotált képkocka hozzáadása a listához
        annotated_frames.append(annotated_frame)

        """
        # Játékosok elipsziseinek rajzolása
        for id, player in enumerate(detected_objects["players"]):
            upper_body_image = teamAssigner.get_upper_body_image(frame, player, id)
            player_color = teamAssigner.get_player_color(upper_body_image, id)
            team_color_num = teamAssigner.get_player_to_team(player_color, team1_color, team2_color, id)
            if team_color_num == 1:
                annotated_frame = draw_ellipse(annotated_frame, player, team1_color, track_id=id)
            elif team_color_num == 2:
                annotated_frame = draw_ellipse(annotated_frame, player, team2_color, track_id=id)"
        """
    
    # Detektált output videó generálása
    generate_output_video(annotated_frames, fps, width, height, output_video_path)

    print(f"Team1 color: {team1_color}")
    print(f"Team2 color: {team2_color}")

if __name__ == "__main__":
    video_path = "input_videos\\szoboszlai.mp4"
    output_video_path = "output_videos\\output_video.avi"
    detect_video(video_path, output_video_path)
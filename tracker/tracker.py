from ultralytics import YOLO
from utils.video_utils import load_video, generate_output_video
from utils.bbox_utils import get_center_of_bbox, get_bbox_width
from utils.team_assigner_utils import TeamAssigner
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
    
    return frame
    
def detect_video(video_path, output_video_path, model):
    
    # Input videó beolvasása
    frames, fps, width, height = load_video(video_path)
    
    annotated_frames = []
    # Detektálás az összes képkockán
    for frame_num, frame in enumerate(frames):
        player_counter = 1
        annotated_frame = frame.copy()

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

        # Elipszis rajzolása a játékosok alá
        for box in boxes_np:
            x1, y1, x2, y2, conf, cls = box
            cls = int(cls)

            # Játékosok
            if cls == 2:
                detected_objects["players"].append([x1, y1, x2, y2])
                bbox = [x1, y1, x2, y2]
                color = (0, 255, 0)
                annotated_frame = draw_ellipse(annotated_frame, bbox, color, track_id=player_counter)
                player_counter += 1

            # Játékvezetők
            elif cls == 3:
                detected_objects["referees"].append([x1, y1, x2, y2])

            # Labda
            elif cls == 0:
                detected_objects["ball"].append([x1, y1, x2, y2])

        annotated_frames.append(annotated_frame)

        # Játékos színének meghatározása
        teamAssigner = TeamAssigner()
        id = 0

        if frame_num == 0:
            # Csapatok színének meghatározása
            team1_color = None
            team2_color = None
            threshold = 70

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
    
    # Detektált output videó generálása
    generate_output_video(annotated_frames, fps, width, height, output_video_path)

    print(f"Team1 color: {team1_color}")
    print(f"Team2 color: {team2_color}")

if __name__ == "__main__":
    video_path = "input_videos\\szoboszlai.mp4"
    output_video_path = "output_videos\\output_video.avi"
    detect_video(video_path, output_video_path)
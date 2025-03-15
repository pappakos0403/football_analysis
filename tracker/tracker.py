from ultralytics import YOLO
from utils.video_utils import load_video, generate_output_video
from utils.bbox_utils import get_center_of_bbox, get_bbox_width
import cv2

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
    for frame in frames:
        player_counter = 1
        annotated_frame = frame.copy()

        results = model(frame)

        # Kapus átállítása játékosra
        # boxes.data tensor oszlopainál a cls átállítása 2-ről 1-re (kapus -> játékos)
        boxes = results[0].boxes
        boxes_data = boxes.data.clone()
        boxes_data.data[boxes.data[:, 5] == 1, 5] = 2
        boxes.data = boxes_data

        # Boxes.data numpy tömbbé alakítása
        boxes_np = boxes.data.cpu().numpy() if hasattr(boxes.data, "cpu") else boxes.data

        # Elipszis rajzolása a játékosok alá
        for box in boxes_np:
            x1, y1, x2, y2, conf, cls = box
            cls = int(cls)

            if cls == 2:
                bbox = [x1, y1, x2, y2]
                color = (0, 255, 0)
                annotated_frame = draw_ellipse(annotated_frame, bbox, color, track_id=player_counter)
                player_counter += 1

        annotated_frames.append(annotated_frame)
            
        """# Annotált képkockák generálása
        annotated_frame = results[0].plot()
        annotated_frames.append(annotated_frame)"""
    
    # Detektált output videó generálása
    generate_output_video(annotated_frames, fps, width, height, output_video_path)

if __name__ == "__main__":
    video_path = "input_videos\\szoboszlai.mp4"
    output_video_path = "output_videos\\output_video.avi"
    detect_video(video_path, output_video_path)
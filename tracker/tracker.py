from ultralytics import YOLO
from utils.video_utils import load_video, generate_output_video

def detect_video(video_path, output_video_path, model):
    
    # Input videó beolvasása
    frames, fps, width, height = load_video(video_path)
    
    annotated_frames = []
    # Detektálás az összes képkockán
    for frame in frames:
        results = model(frame)

        # Kapus átállítása játékosra
        # boxes.data tensor oszlopainál a cls átállítása 2-ről 1-re
        boxes = results[0].boxes
        boxes_data = boxes.data.clone()
        boxes_data.data[boxes.data[:, 5] == 1, 5] = 2
        boxes.data = boxes_data
        
            
        # Annotált képkockák generálása
        annotated_frame = results[0].plot()
        annotated_frames.append(annotated_frame)
    
    # Detektált output videó generálása
    generate_output_video(annotated_frames, fps, width, height, output_video_path)

if __name__ == "__main__":
    video_path = "input_videos\\szoboszlai.mp4"
    output_video_path = "output_videos\\output_video.avi"
    detect_video(video_path, output_video_path)
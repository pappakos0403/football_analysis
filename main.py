from utils import load_video, generate_output_video
from tracker import detect_video

from ultralytics import YOLO

import cv2

# Modell betöltése
model = YOLO("models\\best.pt")

# Detektálás a videón
video_path = "input_videos\\szoboszlai.mp4"
output_video_path = "output_videos\\output_video.avi"
detect_video(video_path, output_video_path, model)

print(model.names)
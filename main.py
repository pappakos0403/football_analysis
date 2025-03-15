from utils import load_video, generate_output_video
from ultralytics import YOLO

import cv2

# Modell betöltése
model = YOLO("models\\best.pt")

# Input videó beolvasása
video_path = ("input_videos\\szoboszlai.mp4")
video_frames, fps, width, height = load_video(video_path)

# Output videó generálása
output_video_path = "output_videos\\output_video.avi"
generate_output_video(video_frames, fps, width, height, output_video_path)
from utils import load_video, generate_output_video
from tracker import detect_video

from ultralytics import YOLO

# Modell betöltése
model = YOLO("models\\best.pt")

# Videófájl elérési útvonalai
video_path = "input_videos\\haaland.mp4"
output_video_path = "output_videos\\output_video.avi"

# Detektálás a videón
detect_video(video_path, output_video_path, model)
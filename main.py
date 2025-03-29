from utils import load_video, generate_output_video
from tracker import Tracker
from ultralytics import YOLO

# Videófájl elérési útvonalai
video_path = "input_videos\\eto.mp4"
output_video_path = "output_videos\\output_video.avi"
stub_path = "stubs\\eto.pkl"

# Modell betöltése
tracker = Tracker("models\\best.pt")

# Videó betöltése
frames, fps, width, height = load_video(video_path)

# Detektálás a videón
annotated_frames, fps, width, height, = tracker.detect_video(frames, fps, width, height, read_from_stub=True, stub_path=stub_path)

# Output videó generálása
generate_output_video(annotated_frames, fps, width, height, output_video_path)
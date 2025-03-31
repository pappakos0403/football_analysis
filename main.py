from utils import load_video, generate_output_video
from tracker import Tracker
from ultralytics import YOLO

# Videófájl elérési útvonalai
video_path = "input_videos\\08fd33_4.mp4"
output_video_path = "output_videos\\output_video.avi"
stub_path = "stubs\\08fd33_4.pkl"

# Modell betöltése
tracker = Tracker("models\\best.pt")

# Videó betöltése
frames, fps, width, height = load_video(video_path)

# Detektálás a videón
tracks = tracker.detect_video(frames, read_from_stub=True, stub_path=stub_path)

# Annotálás a videón
annotated_frames = tracker.annotations(frames, tracks)

# Output videó generálása
generate_output_video(annotated_frames, output_video_path, fps, width, height)
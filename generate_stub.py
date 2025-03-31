import os
import pickle
from utils import load_video
from tracker import Tracker

# Útvonalak
video_path = "input_videos/08fd33_4.mp4"
stub_path = "stubs/08fd33_4.pkl"
model_path = "models/best.pt"

# Stub mappa létrehozása, ha nem létezik
os.makedirs("stubs", exist_ok=True)

# Modell és videó betöltése
tracker = Tracker(model_path)
frames, fps, width, height = load_video(video_path)

# Detektálás és stub mentés
tracks = tracker.detect_video(frames, read_from_stub=False, stub_path=None)

# Stub fájl mentése
with open(stub_path, "wb") as f:
    pickle.dump(tracks, f)
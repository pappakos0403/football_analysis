import os
import pickle
from utils import load_video
from tracker import Tracker

# Útvonalak
video_path = "input_videos/eto.mp4"
stub_path = "stubs/eto.pkl"
model_path = "models/best.pt"

# Stub mappa létrehozása, ha nem létezik
os.makedirs("stubs", exist_ok=True)

# Modell és videó betöltése
tracker = Tracker(model_path)
frames, fps, width, height = load_video(video_path)

# Detektált annotált frame-ek generálása
annotated_frames, fps, width, height = tracker.detect_video(frames, fps, width, height)

# Stub fájl mentése
with open(stub_path, "wb") as f:
    pickle.dump(annotated_frames, f)
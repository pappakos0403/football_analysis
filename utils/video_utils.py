import cv2
import subprocess
import os

# Input videó beolvasása
def load_video(path):
    cap = cv2.VideoCapture(path)

    # Videó beolvasása frame-ekbe
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return frames, fps, width, height

# Output videó generálása
def generate_output_video(output_video_frames, output_video_path, video_fps, video_width, video_height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (video_width, video_height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
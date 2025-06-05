import cv2
from pathlib import Path

# Thumbnail kép mentése a videóból
def save_video_thumbnail(source_video_path: str, output_video_path: str):
    source_video_path = Path(source_video_path)
    output_video_path = Path(output_video_path)

    thumbnail_folder = output_video_path.parent / "thumbnail"
    thumbnail_folder.mkdir(parents=True, exist_ok=True)
    thumbnail_path = thumbnail_folder / "thumbnail.jpg"

    cap = cv2.VideoCapture(str(source_video_path))
    success, frame = cap.read()
    cap.release()

    if success:
        cv2.imwrite(str(thumbnail_path), frame)
        return str(thumbnail_path)
    else:
        return None

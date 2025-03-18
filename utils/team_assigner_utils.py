from sklearn.cluster import KMeans
import cv2

def get_upper_body_image(frame, bbox):
    # A frame adott játékosának felső testét tartalmazó kép kivágása
    x1, y1, x2, y2, = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Bbox magassága
    bbox_height = y2 - y1

    # Bbox felső részének végpontja
    upper_body_y = y1 + int(0.5 * bbox_height)

    # Felső test
    upper_body_image = frame[y1:upper_body_y, x1:x2]

    # Felső test kép mentése
    cv2.imwrite("test_image\\upper_body.jpg", upper_body_image)

def get_clustering_model(image):
    # Kép átalakítása 2D-s tömbbé
    image_2d = image.reshape(-1, 3)

    # K-means modell létrehozása és tanítása
    kmeans = KMeans(n_clusters=2, init="k-means++", random_state=0)
    kmeans.fit(image_2d)

    return kmeans
from sklearn.cluster import KMeans
import cv2
import numpy as np

class TeamAssigner:

    def get_upper_body_image(self, frame, bbox, id):
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
        cv2.imwrite(f"test_image\\upper_body{id}.jpg", upper_body_image)

        return upper_body_image

    def get_clustering_model(self, image):
        
        # Kép átalakítása 2D-s tömbbé
        image_2d = image.reshape(-1, 3)

        # K-means modell létrehozása és tanítása
        kmeans = KMeans(n_clusters=2, init="k-means++", random_state=0)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, image, id):
        # Klaszterező modell létrehozása a képből
        kmeans = self.get_clustering_model(image)

        # Pixelek klaszterének meghatározása
        labels = kmeans.labels_

        #  Klaszterezett kép létrehozása
        clustered_image = labels.reshape(image.shape[0], image.shape[1])

        clustered_image_scaled = (clustered_image * 255).astype(np.uint8)
        colored_clustered_image = cv2.applyColorMap(clustered_image_scaled, cv2.COLORMAP_JET)
        cv2.imwrite(f"test_image\\clustered_image{id}.jpg", colored_clustered_image)
        id += 1


        # Klaszterezett kép sarkainak meghatározása
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], 
                           clustered_image[-1, 0], clustered_image[-1, -1]]
        
        # A játékos klaszter meghatározása
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # Mezszín meghatározása
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color
    
    def get_player_to_team(self, color, team1_color, team2_color, id):
        # Euklideszi távolság kiszámítása mindkét színhez
        dist_to_team1 = np.linalg.norm(np.array(team1_color) - np.array(color))
        dist_to_team2 = np.linalg.norm(np.array(team2_color) - np.array(color))

        # Játékos besorolása a csapatába
        if dist_to_team1 <= dist_to_team2:
            return 1
        else:
            return 2
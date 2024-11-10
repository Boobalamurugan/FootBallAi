from sklearn.cluster import KMeans
import numpy as np

class TeamClassifier:
    def __init__(self, n_clusters=2):
        """
        Initializes the TeamClassifier with the number of clusters (teams).
        
        Args:
            n_clusters (int): The number of teams (default is 2).
        """
        self.n_clusters = n_clusters
        self.team_colors = {}

    def _get_clustering_model(self, image):
        """
        Applies KMeans clustering on the image to find dominant colors.
        
        Args:
            image (np.ndarray): The input image (2D or 3D array).
        
        Returns:
            KMeans: A fitted KMeans model.
        """
        # Flatten the image into 2D (pixels as rows, color channels as columns)
        img_2d = image.reshape(-1, 3)
        
        # Apply KMeans clustering to the pixels in the image
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=20).fit(img_2d)
        return kmeans

    def _get_player_color(self, frame, bbox):
        """
        Extracts the most dominant color from the upper half of a player's bounding box.
        
        Args:
            frame (np.ndarray): The current video frame.
            bbox (list or tuple): The bounding box of the player (x1, y1, x2, y2).
        
        Returns:
            np.ndarray: The color of the player (RGB).
        """
        # Crop the region of interest (player's bounding box)
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        # Focus on the top half of the player's cropped image to detect shirt color
        top_half_img = image[0:int(image.shape[0] // 2), :]
        
        # Get KMeans clustering model for the top half of the image
        kmeans = self._get_clustering_model(top_half_img)
        
        # Get cluster labels for each pixel in the top half image
        labels = kmeans.labels_
        
        # Reshape labels to match the image's shape
        clustered_img = labels.reshape(top_half_img.shape[0], top_half_img.shape[1])
        
        # Determine the dominant cluster using the corners of the image
        corner_cluster = [
            clustered_img[0, 0], clustered_img[0, -1],
            clustered_img[-1, 0], clustered_img[-1, -1]
        ]
        
        # Identify the non-player cluster (background or non-relevant region)
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        
        # The remaining cluster is the player's shirt color
        player_cluster = 1 - non_player_cluster
        
        # Get the color of the player's cluster
        player_color = kmeans.cluster_centers_[player_cluster]
        
        return player_color

    def classify_players(self, frame, player_detections):
        """
        Classifies players into teams based on the dominant color of their shirts.
        
        Args:
            frame (np.ndarray): The current video frame.
            player_detections (object): An object containing player detections with bbox and tracker_id.
        
        Returns:
            dict: A dictionary containing the team colors.
            dict: A dictionary mapping player IDs to team assignments.
        """
        player_colors = []
        player_ids = []

        # Extract player colors and IDs from the detections
        for bbox, tracker in zip(player_detections.xyxy, player_detections.tracker_id):
            player_color = self._get_player_color(frame, bbox)
            player_colors.append(player_color)
            player_ids.append(tracker)

        # If there are fewer than two players, return empty dictionaries (no teams to classify)
        if len(player_colors) < 2:
            return {}, {}

        # Apply KMeans clustering to group players based on shirt color
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=50, max_iter=300)
        kmeans.fit(player_colors)

        # Save the team colors (cluster centers)
        self.team_colors = {
            1: kmeans.cluster_centers_[0],
            2: kmeans.cluster_centers_[1]
        }

        # Assign each player to the team with the closest color
        team_assignments = {}
        for player_id, color in zip(player_ids, player_colors):
            # Calculate the Euclidean distance between the player's color and each team's color
            distances = {team_id: np.linalg.norm(color - self.team_colors[team_id]) for team_id in self.team_colors}
            
            # Assign player to the closest team
            closest_team = min(distances, key=distances.get)
            team_assignments[player_id] = closest_team

        return self.team_colors, team_assignments

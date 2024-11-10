import numpy as np
import cv2

class ViewTransformer:
    """
    A class for transforming 2D points between two coordinate spaces using homography.

    This can be useful for transforming points from one view (e.g., source camera view) 
    to another (e.g., target view) using a homography transformation matrix.

    Parameters:
    - source (np.ndarray): An array of shape (N, 2) containing the coordinates of points 
      in the source space.
    - target (np.ndarray): An array of shape (N, 2) containing the coordinates of points 
      in the target space.

    Raises:
    - ValueError: If the source and target arrays have mismatched shapes, or if they do 
      not represent 2D coordinates.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")

        # Convert source and target points to float32 for OpenCV compatibility
        source = source.astype(np.float32)
        target = target.astype(np.float32)

        # Calculate the homography matrix between source and target points
        self.m, _ = cv2.findHomography(source, target)
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transforms a set of 2D points using the homography matrix.

        Parameters:
        - points (np.ndarray): An array of shape (M, 2) containing the 2D points to be 
          transformed.

        Returns:
        - np.ndarray: An array of transformed 2D points with the same shape as the input.

        Raises:
        - ValueError: If the points array is not 2D coordinates.
        """
        if points.size == 0:
            # Return an empty array if no points are provided
            return points

        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        # Reshape points for perspective transformation and apply the transformation
        points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(points, self.m)
        return transformed_points.reshape(-1, 2).astype(np.float32)

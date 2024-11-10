from typing import Optional, List
import cv2
import supervision as sv
import numpy as np
from ..config.football_pitch_config import FootBallPitchConfiguration

def draw_pitch(
    config: FootBallPitchConfiguration,
    background_color: sv.Color = sv.Color(34, 139, 34),  # Default green pitch
    line_color: sv.Color = sv.Color.WHITE,  # Default line color is white
    padding: int = 50,  # Padding around the pitch
    line_thickness: int = 4,  # Thickness of lines on the pitch
    point_radius: int = 8,  # Radius for circular points (e.g., penalty spots)
    scale: float = 0.1  # Scaling factor to control pitch size
) -> np.ndarray:
    """
    Draws a football pitch using a given configuration.

    Parameters:
    - config: FootBallPitchConfiguration object with pitch dimensions and vertices.
    - background_color: Color of the pitch background.
    - line_color: Color of the pitch lines.
    - padding: Padding around the pitch border.
    - line_thickness: Thickness of the lines drawn on the pitch.
    - point_radius: Radius for key circular points (e.g., penalty spots).
    - scale: Scaling factor for the pitch dimensions.

    Returns:
    - pitch_image: Image array of the drawn pitch.
    """
    # Scale pitch dimensions based on the given scale factor
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    # Create a blank image with a background color
    pitch_image = np.ones(
        (scaled_width + 2 * padding, scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    # Draw lines for pitch edges using the vertex connections specified in config
    for start, end in config.edges:
        point1 = (
            int(config.vertices[start - 1][0] * scale) + padding,
            int(config.vertices[start - 1][1] * scale) + padding
        )
        point2 = (
            int(config.vertices[end - 1][0] * scale) + padding,
            int(config.vertices[end - 1][1] * scale) + padding
        )
        cv2.line(
            img=pitch_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

    # Draw the centre circle
    centre_circle_center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding
    )
    cv2.circle(
        img=pitch_image,
        center=centre_circle_center,
        radius=scaled_circle_radius,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )

    # Draw penalty spots
    penalty_spots = [
        (scaled_penalty_spot_distance + padding, scaled_width // 2 + padding),
        (scaled_length - scaled_penalty_spot_distance + padding, scaled_width // 2 + padding)
    ]
    for spot in penalty_spots:
        cv2.circle(
            img=pitch_image,
            center=spot,
            radius=point_radius,
            color=line_color.as_bgr(),
            thickness=-1  # Solid circle
        )

    return pitch_image

def draw_points_on_pitch(
    config: FootBallPitchConfiguration,
    xy: np.ndarray,  # Array of points to draw
    face_color: sv.Color = sv.Color.RED,  # Fill color for points
    edge_color: sv.Color = sv.Color.BLACK,  # Border color for points
    radius: int = 10,  # Radius of points
    thickness: int = 2,  # Thickness of the border
    padding: int = 50,  # Padding around the pitch
    scale: float = 0.1,  # Scaling factor
    pitch: Optional[np.ndarray] = None  # Optionally pass a pre-drawn pitch
) -> np.ndarray:
    """
    Draws points on the football pitch.

    Parameters:
    - config: FootBallPitchConfiguration object with pitch dimensions.
    - xy: Numpy array of points (coordinates).
    - face_color: Color for the filled circle.
    - edge_color: Color for the circle border.
    - radius: Radius of the circles representing points.
    - thickness: Thickness of the circle border.
    - padding: Padding around the pitch border.
    - scale: Scaling factor for points.
    - pitch: Optional pre-drawn pitch image.

    Returns:
    - pitch: Image array with points drawn on the pitch.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    for point in xy:
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        # Draw filled circle (face)
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1
        )
        # Draw circle border (edge)
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness
        )

    return pitch

import cv2
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
from ..annotator.football_pitch_annotator import draw_pitch, draw_points_on_pitch
from ..annotator.view_transform import ViewTransformer
from ..config.football_pitch_config import FootBallPitchConfiguration


class KeyPointDetection:
    def __init__(self, model_path: str):
        # Initialize the YOLO model and pitch configuration
        self.model = YOLO(model_path)
        self.config = FootBallPitchConfiguration()

    def preprocess_frames(self, video_frames: List[np.ndarray]) -> Tuple[int, int, List[np.ndarray]]:
        # Extract frame dimensions and compute scaling factors based on 640x640 input size
        frame_width, frame_height = video_frames[0].shape[1], video_frames[0].shape[0]
        scale_x, scale_y = frame_width / 640, frame_height / 640
        return frame_width, frame_height, scale_x, scale_y

    def assign_players_to_teams(self, players_detections: sv.Detections, team_assignments: Dict[int, int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # Assign players to teams (team A or B) based on their IDs
        team_A_xy = []
        team_B_xy = []
        players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        for detection_idx, player_id in enumerate(players_detections.tracker_id):
            team_id = team_assignments.get(player_id)
            player_xy = players_xy[detection_idx]
            if team_id == 1:
                team_A_xy.append(player_xy)
            elif team_id == 2:
                team_B_xy.append(player_xy)
        return team_A_xy, team_B_xy

    def draw_points_on_pitch(self, annotated_frame: np.ndarray, pitch_points: np.ndarray, color: sv.Color, radius: int) -> np.ndarray:
        # Draw points on the football pitch (e.g., player positions, ball position)
        return draw_points_on_pitch(
            config=self.config,
            xy=pitch_points,
            face_color=color,
            edge_color=sv.Color.BLACK,
            radius=radius,
            pitch=annotated_frame
        )

    def overlay_pitch_on_frame(self, annotated_frame: np.ndarray, frame: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
        # Overlay the football pitch onto the frame at a specific position
        pitch_height, pitch_width = int(frame_height * 0.4), int(frame_width * 0.5)
        pitch_resized = cv2.resize(annotated_frame, (pitch_width, pitch_height))
        pitch_y = frame_height - pitch_height
        pitch_x = (frame_width - pitch_width) // 2
        alpha = 0.5
        overlay = frame[pitch_y:pitch_y + pitch_height, pitch_x:pitch_x + pitch_width]
        cv2.addWeighted(pitch_resized, alpha, overlay, 1 - alpha, 0, overlay)
        return frame

    def get_keypoint_detection(self, video_frames: List[np.ndarray],
                                detections: List[sv.Detections], 
                                team_assignments: Dict[int, int]) -> List[np.ndarray]:
        # Main function to detect key points (ball, players, goalkeepers, referees) and assign teams
        frame_width, frame_height, scale_x, scale_y = self.preprocess_frames(video_frames)
        detected_frames = []

        for frame, detection in tqdm(zip(video_frames, detections),
                                     desc="KeyPoint Detection Processing",
                                     unit="frame",
                                     total=len(video_frames)):
            # Resize the frame for processing
            frame_resize = cv2.resize(frame, (640, 640))
            result = self.model(frame_resize)[0]
            key_points = sv.KeyPoints.from_ultralytics(result)
            
            # Filter points with confidence greater than 0.5
            filter = key_points.confidence[0] > 0.5
            frame_reference_points = key_points.xy[0][filter]
            pitch_reference_points = np.array(self.config.vertices)[filter]
            
            # Scale frame reference points for proper transformation
            frame_reference_points *= np.array([[scale_x, scale_y]])

            # Initialize view transformer for coordinate mapping
            transformer = ViewTransformer(
                source=frame_reference_points,
                target=pitch_reference_points
            )

            # Ball detection: Extract ball coordinates and transform them to pitch coordinates
            ball_detections = detection[detection.class_id == 0]
            frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_ball_xy = transformer.transform_points(frame_ball_xy)

            # Player detection: Assign players to teams and transform their coordinates
            players_detections = detection[detection.class_id == 2]
            team_A_xy, team_B_xy = self.assign_players_to_teams(players_detections, team_assignments)
            pitch_team_A_xy = transformer.transform_points(np.array(team_A_xy))
            pitch_team_B_xy = transformer.transform_points(np.array(team_B_xy))

            # Goalkeeper detection: Transform goalkeeper coordinates
            goalkeeper_detections = detection[detection.class_id == 1]
            goalkeeper_xy = goalkeeper_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_goalkeeper_xy = transformer.transform_points(goalkeeper_xy)

            # Referee detection: Transform referee coordinates
            referees_detections = detection[detection.class_id == 3]
            referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_referees_xy = transformer.transform_points(referees_xy)

            # Create an empty annotated frame and draw all detected points
            annotated_frame = draw_pitch(self.config)
            annotated_frame = self.draw_points_on_pitch(annotated_frame, pitch_ball_xy, sv.Color.WHITE, 10)
            annotated_frame = self.draw_points_on_pitch(annotated_frame, pitch_team_A_xy, sv.Color.from_rgb_tuple((250, 4, 193)), 16)
            annotated_frame = self.draw_points_on_pitch(annotated_frame, pitch_team_B_xy, sv.Color.from_rgb_tuple((4, 201, 250)), 16)
            annotated_frame = self.draw_points_on_pitch(annotated_frame, pitch_goalkeeper_xy, sv.Color.BLUE, 16)
            annotated_frame = self.draw_points_on_pitch(annotated_frame, pitch_referees_xy, sv.Color.from_hex('FFD700'), 16)

            # Overlay the pitch with the detected points onto the frame
            frame = self.overlay_pitch_on_frame(annotated_frame, frame, frame_width, frame_height)

            detected_frames.append(frame)

        return detected_frames

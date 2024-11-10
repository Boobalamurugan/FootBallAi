import pickle 
from ultralytics import YOLO
from tqdm import tqdm
import supervision as sv
import os 

def get_detection(model_path, video_frames, stubs_path=None, stubs_status=False):
    """
    Detects and tracks objects in video frames using a YOLO model and ByteTrack tracker.
    
    Parameters:
    - model_path (str): Path to the YOLO model file.
    - video_frames (list): List of video frames (images) to be processed.
    - stubs_path (str, optional): File path to save or load detection results for caching.
    - stubs_status (bool): Whether to load cached detection results if available.
    
    Returns:
    - detected_frames (list): List of tracked detections for each frame.
    """
    # Check if we should load detections from a cached file
    if stubs_status and stubs_path is not None and os.path.exists(stubs_path):
        with open(stubs_path, "rb") as f:
            detections = pickle.load(f)
        return detections
    
    # Load YOLO model for detection
    model = YOLO(model=model_path)
    byte_tracker = sv.ByteTrack()  # Initialize tracker for associating detections across frames
    detected_frames = []

    # Use tqdm to show progress for detection processing
    for frame in tqdm(video_frames, desc="Processing Frames", unit="frame"):
        # Run detection and tracking on the current frame
        result = model.track(frame, conf=0.2)[0]
        detected_frame = sv.Detections.from_ultralytics(result)
        tracked_detections = byte_tracker.update_with_detections(detections=detected_frame)
        detected_frames.append(tracked_detections)
    
    # Optionally save detections to a file for later use
    if stubs_path is not None:
        with open(stubs_path, "wb") as f:
            pickle.dump(detected_frames, f)
    
    return detected_frames

def draw_annotation(video_frames, detection, team_assignments, assigned_players):
    """
    Annotates video frames with detected objects using customized shapes and labels.
    
    Parameters:
    - video_frames (list): List of video frames to annotate.
    - detection (list): Detection results for each frame.
    - team_assignments (dict): Dictionary mapping player IDs to team numbers (e.g., 1 or 2).
    - assigned_players (list): List of player IDs that have the ball assigned for each frame.
    
    Returns:
    - annotated_frames (list): List of annotated video frames.
    """
    annotated_frames = []
    
    # Define annotators for different object types (players, referees, ball)
    annotators = {
        1: {  # Team 1
            'ellipse': sv.EllipseAnnotator(color=sv.Color.from_rgb_tuple((250, 4, 193)), 
                                           thickness=2),
            'label': sv.LabelAnnotator(
                color=sv.Color.from_rgb_tuple((250, 4, 193)),
                text_color=sv.Color.BLACK,
                text_scale=0.4,
                text_thickness=1,
                text_padding=10,
                text_position=sv.Position.BOTTOM_CENTER
            )
        },
        2: {  # Team 2
            'ellipse': sv.EllipseAnnotator(color=sv.Color.from_rgb_tuple((4, 201, 250)),
                                           thickness=2),
            'label': sv.LabelAnnotator(
                color=sv.Color.from_rgb_tuple((4, 201, 250)),
                text_color=sv.Color.BLACK,
                text_scale=0.4,
                text_thickness=1,
                text_padding=10,
                text_position=sv.Position.BOTTOM_CENTER
            )
        },
        'referee': {  # Referee annotation
            'ellipse': sv.EllipseAnnotator(color=sv.Color.from_hex('#FFD700'), 
                                           thickness=2)

        },
        'goalkeeper': {  # goalkeeper annotation
            'ellipse': sv.EllipseAnnotator(color=sv.Color.BLUE, 
                                           thickness=2),
        },
        'ball': sv.TriangleAnnotator(  # Ball annotation
            color=sv.Color.from_hex('#FFD700'),
            base=25,
            height=21,
            outline_thickness=1
        ),
        'marker': sv.TriangleAnnotator(  # Marker for player with the ball
            color=sv.Color.from_hex('#ff0000'),
            base=25,
            height=21,
            outline_thickness=1,
            position=sv.Position.TOP_CENTER
        )
    }
    
    # Iterate over video frames and corresponding detections
    for frame_num, frame in tqdm(enumerate(video_frames), total=len(video_frames),
                                 desc="Annotating frames"):
        frame = frame.copy()  # Create a copy of the frame for annotation

        # Filter detections for the current frame
        ball_detection = detection[frame_num][detection[frame_num].class_id == 0]  # Ball detections
        player_detection = detection[frame_num][detection[frame_num].class_id == 2]  # Player detections
        referee_detection = detection[frame_num][detection[frame_num].class_id == 3]  # Referee detections
        goalkeeper_detection = detection[frame_num][detection[frame_num].class_id == 1]  

        # Determine which player is assigned to the ball
        assigned_player = assigned_players[frame_num] if frame_num < len(assigned_players) else None

        # Annotate ball with a triangle marker and padding
        if len(ball_detection) > 0:
            padded_box = sv.pad_boxes(xyxy=ball_detection.xyxy, px=10)  # Add 10px padding to box
            ball_detection.xyxy = padded_box  # Update detection box
            frame = annotators['ball'].annotate(scene=frame, detections=ball_detection)

        # Annotate players based on team assignment
        for detection_idx, player_id in enumerate(player_detection.tracker_id):
            team_id = team_assignments.get(player_id)  # Get team assignment
            if team_id in annotators:  # Check if player belongs to a valid team
                player_det = player_detection[detection_idx:detection_idx + 1]  # Extract detection
                frame = annotators[team_id]['ellipse'].annotate(scene=frame, 
                                                                detections=player_det)
                frame = annotators[team_id]['label'].annotate(scene=frame,
                                                              detections=player_det,
                                                              labels=[f'#{player_id}'])
            
            # Annotate player with the ball using a marker
            if player_id == assigned_player:
                padded_box = sv.pad_boxes(xyxy=player_det.xyxy, px=10)  # Add padding
                player_det.xyxy = padded_box  # Update detection box
                frame = annotators['marker'].annotate(scene=frame, 
                                                      detections=player_det)
        
        # Annotate referees
        if len(referee_detection) > 0:
            frame = annotators['referee']['ellipse'].annotate(scene=frame, 
                                                            detections=referee_detection)

        if len(goalkeeper_detection) > 0:
            frame = annotators['goalkeeper']['ellipse'].annotate(scene=frame, 
                                                            detections=goalkeeper_detection)
        # Append the annotated frame to the list
        annotated_frames.append(frame)

    return annotated_frames

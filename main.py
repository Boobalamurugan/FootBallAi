import torch
from tqdm import tqdm
from src.utils.team_classifer import TeamClassifier
from src.keypoint_detection.keypoints import KeyPointDetection
from src.utils.video_utils import read_video, save_video
from src.utils.ball_acquisition import assign_ball_to_player
from src.detections.player_detection import draw_annotation, get_detection


# Clear any existing CUDA cache to optimize memory usage
# torch.cuda.empty_cache()

# Global variables for paths
VIDEO_PATH = "./test_videos/five.mp4"
OBJ_MODEL_PATH = "./models/obj_v8x.pt"
POSE_MODEL_PATH = "./models/pose_v8x.pt"
OUTPUT_PATH = "./output_videos/five_obj.mp4"
STUBS_PATH_OBJ = "./stub/five_obj.pkl"
STUBS_STATUS = False

def main():

    """Main function to handle the end-to-end football video analysis pipeline."""

    # Step 1: Read video frames from the input video path
    video_frames = read_video(VIDEO_PATH)

    # Step 2: Perform object detection on video frames using YOLO model
    detections = get_detection(OBJ_MODEL_PATH, video_frames, STUBS_PATH_OBJ, STUBS_STATUS)

    # Step 3: Initialize a team classifier for player classification
    team_classifier = TeamClassifier()
    goalkeeper_id = None  # Initialize goalkeeper ID to track later

    # Step 4: Classify players and assign them to teams
    for frame_num, frame in tqdm(enumerate(video_frames), total=len(video_frames), desc="Clustering Process", unit="frame"):
        player_detections = detections[frame_num][detections[frame_num].class_id == 2]  # Filter only player detections

        if len(player_detections) > 0:
            # Classify players into teams based on detections
            _, team_assignments = team_classifier.classify_players(frame, player_detections)

            # Detect and assign goalkeeper ID if not already assigned
            if goalkeeper_id is None:
                goalkeepers = detections[frame_num][detections[frame_num].class_id == 1]
                if len(goalkeepers) > 0:
                    goalkeeper_id = goalkeepers.tracker_id[0]  # Assuming first detected as the goalkeeper

    # Step 5: Assign the ball to the nearest player in each frame
    assigned_players = []
    for frame_num, detection in tqdm(enumerate(detections), total=len(detections), desc="Ball Acquisition"):
        player_detections = detection[detection.class_id == 2]  # Filter player detections
        ball_detections = detection[detection.class_id == 0]    # Filter ball detections

        # Prepare player data for ball assignment
        players = {
            player_id: {"bbox": player_bbox}
            for player_id, player_bbox in zip(player_detections.tracker_id, player_detections.xyxy.tolist())
        }

        if len(ball_detections) > 0:
            for ball_bbox in ball_detections.xyxy:
                # Assign the ball to the nearest player
                assigned_player = assign_ball_to_player(players, ball_bbox)
                assigned_players.append(assigned_player)
        else:
            # No ball detected in this frame
            assigned_players.append(-1)

    # Step 6: Perform keypoint detection with team assignments
    keypoint_detections = KeyPointDetection(POSE_MODEL_PATH)
    keypoint_detected_frames = keypoint_detections.get_keypoint_detection(
        video_frames=video_frames,
        detections=detections,
        team_assignments=team_assignments
    )

    # Step 7: Annotate frames and save the annotated video to the output path
    annotated_frames = draw_annotation(
        keypoint_detected_frames,
        detections,
        team_assignments,
        assigned_players
    )
    
    save_video(annotated_frames, OUTPUT_PATH)


if __name__ == "__main__":
    main()

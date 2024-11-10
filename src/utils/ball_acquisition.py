from .video_utils import get_center_of_bbox, measure_distance

def assign_ball_to_player(players, ball_bbox):
    """
    Assigns a player to the ball based on the distance between the player and the ball.

    Args:
        players (dict): A dictionary of players where each key is a player ID and the value is a dictionary 
                        containing player details (e.g., "bbox" for the bounding box of the player).
        ball_bbox (np.ndarray): The bounding box of the ball.

    Returns:
        int: The ID of the assigned player. If no player is close enough, returns -1.
    """
    
    # Get the center position of the ball using its bounding box
    ball_position = get_center_of_bbox(ball_bbox)
    
    # Set the maximum distance at which a player can be assigned to the ball
    max_player_ball_distance = 70
    
    # Initialize minimum distance as infinity and assigned player as -1 (none assigned)
    min_distance = float('inf')
    assigned_player = -1

    # Loop through each player to check the distance to the ball
    for player_id, player in players.items():
        player_bbox = player["bbox"]
        
        # Get the center position of the player's bounding box
        player_position = get_center_of_bbox(player_bbox)
        
        # Calculate the distance between the player's center and the ball's center
        distance = measure_distance(player_position, ball_position)

        # If the player is within the allowed distance and closer than previous players
        if distance < max_player_ball_distance and distance < min_distance:
            min_distance = distance
            assigned_player = player_id

    # Return the ID of the assigned player, or -1 if no assignment was made
    return assigned_player

# FootBallAi

**FootBallAi** is an AI-driven football analysis system designed for player detection, tracking, and classification in real-time video streams. This project uses YOLO (You Only Look Once) for object detection, KMeans for team color assignment, and advanced techniques for player tracking and keypoint detection. It aims to enhance football analytics by automating tasks such as player detection, player tracking, and ball assignment.

## Key Features

- **YOLO Object Detection**: Detects football players and the ball in real-time using YOLO.
- **Player Tracking**: Tracks player movements across frames to provide insights into their positions.
- **Team Classification**: Classifies players based on their team using KMeans clustering for color assignment.
- **Keypoint Detection**: Uses keypoints for precise player positioning and movement analysis.
- **Ball Assignment**: Assigns the ball to the closest player and tracks its location.
- **Homography for Player Location**: Computes player locations using homography transformations.
- **Video Processing**: Processes football match videos and analyzes events.

## Future Improvements

- **Enhanced Ball Detection**: Improve ball detection accuracy and training.
- **Performance Optimization**: Reduce processing time to handle real-time analysis more efficiently.
- **Speed and Distance Calculations**: Incorporate calculations for player speed and distance traveled.
- **Siglip Integration**: Upgrade the system with techniques like Siglip for advanced tracking.

## Installation

### Requirements

- Python 3.7 or higher
- YOLOv5/YOLOv8 (depending on the version used)
- OpenCV
- NumPy
- KMeans
- Matplotlib
- PyTorch (depending on YOLO version)

### Dependencies

1. Clone the repository:
   ```bash
   git clone https://github.com/Boobalamurugan/FootBallAi.git
   cd footballAI
   ```
2. Create a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate  # For Linux/macOS
    env\Scripts\activate  # For Windows
    ```
3.Install dependencies:  
    ```bash
    pip install -r requirements.txt
    ```
4.Download pre-trained YOLO weights inside the models folder from the provided text file drive link.

## Contribution

Contributions are welcome! Feel free to fork this project and submit pull requests.

## Acknowledgments

- Thanks to **Piotr Skalski** for his valuable tutorial.
- The **YOLO** algorithm and its developers.
- OpenCV, KMeans, and other open-source libraries used in the project.



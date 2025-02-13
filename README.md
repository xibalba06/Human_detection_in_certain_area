# Human_detection_in_certain_area
# YOLOv5 Human Detection with Boundary Check

This project uses the YOLOv5 object detection model to detect humans in a video stream (e.g., from a laptop camera) and checks whether the detected humans are inside a predefined boundary box. The results are displayed in real-time, and the processed video is saved to a file.

---

## Features

- Real-time human detection using YOLOv5.
- Boundary box check to determine if detected humans are inside a specific area.
- Saves the processed video to a file (`output.avi`).
- Displays the video stream with detection results in real-time.

---

## Requirements

- Python 3.7 or higher
- A laptop or external camera
- CUDA-enabled GPU (optional, for faster processing)

---

## Installation

Follow these steps to set up and run the project:

### 1. Clone the Repository
''' bash
git clone https://github.com/your-username/Human_detection_in_certain_area.git
cd Human_detection_in_certain_area

2. Install Dependencies
Make sure you have Python installed. Then, install the required Python libraries:

pip install -r requirements.txt
The requirements.txt file includes the following dependencies:

torch (PyTorch for running the YOLOv5 model)
torchvision (for PyTorch utilities)
numpy (for numerical operations)
opencv-python (for video processing)
ultralytics (for YOLOv5 model loading)

3. Verify PyTorch Installation
If you have a CUDA-enabled GPU, ensure that PyTorch is installed with GPU support. You can verify this by running:
cuda_tester.py in this repositoies


1. Run the Script
To start the human detection script, run:
python your_script_name.py

2. Key Features
The script will open your laptop camera and start detecting humans in real-time.
A white rectangle represents the predefined boundary box.
Detected humans inside the boundary box are marked with a blue rectangle, while those outside are marked with a red rectangle.
The processed video is saved as output.avi in the project directory.

3. Exit the Program
Press the q key to stop the program and close the video stream.

File Structure
.
├── your_script_name.py       # Main Python script
├── requirements.txt          # List of dependencies
├── README.md                 # Project documentation
└── output.avi                # Processed video (generated after running the script)

Customization
1. Change the Boundary Box
You can modify the boundary box by changing the BOUNDARY_BOX variable in the script:

BOUNDARY_BOX = (100, 100, 500, 400)  # (x1, y1, x2, y2)
2. Adjust the Camera Source
By default, the script uses the laptop camera (cv2.VideoCapture(0)). To use an external camera, change the 0 to the appropriate camera index.

Example Output
When the script is running, you will see a real-time video feed with the following features:

White Rectangle: The predefined boundary box.
Blue Rectangle: Detected humans inside the boundary box.
Red Rectangle: Detected humans outside the boundary box.
The processed video will also be saved as output.avi.

Troubleshooting
1. PyTorch GPU Issues
If PyTorch is not using your GPU, ensure that:

You have installed the correct version of PyTorch with CUDA support.
Your GPU drivers are up to date.
2. OpenCV Camera Issues
If the camera does not open, ensure that:

Your camera is connected and not being used by another application.
You have the correct camera index in cv2.VideoCapture().
Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
YOLOv5 by Ultralytics
PyTorch
OpenCV

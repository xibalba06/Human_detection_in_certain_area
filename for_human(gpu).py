import cv2
import torch
import numpy as np

# Load the YOLO model and specify the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load("ultralytics/yolov5", "yolov5s").to(device)  # Small YOLOv5 model

# Check which device the model is running on
print(f"Model is running on: {next(model.model.parameters()).device}")

# YOLO class for the human category
HUMAN_CLASS = 0  # "person" class in the COCO dataset

# Define a specific area (e.g., a rectangular region)
BOUNDARY_BOX = (100, 100, 500, 400)  # (x1, y1, x2, y2)

def is_inside_boundary(x, y, boundary):
    """Check if a point is inside a specific area."""
    x1, y1, x2, y2 = boundary
    return x1 <= x <=2 and y1 <= y <=2

# Start the laptop camera
cap = cv2.VideoCapture(0)

# Video recorder settings
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Default to 30 FPS if FPS value is unavailable
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video codec (XVID format)
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))  # Create the video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the image for input to the YOLO model
    # Convert the image from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run the model (AutoShape allows direct use of NumPy arrays)
    results = model(frame_rgb)

    # Get the detected objects
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]

    # Draw the defined area on the image
    x1, y1, x2, y2 = BOUNDARY_BOX
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)  # White rectangle

    for *box, conf, cls in detections:
        if int(cls) == HUMAN_CLASS:  # Check only for humans
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Check if inside or outside the defined area
            if is_inside_boundary(center_x, center_y, BOUNDARY_BOX):
                color = (255, 0, 0)  # Blue
            else:
                color = (0, 0, 255)  # Red

            # Draw the detected box and its center
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            #cv2.circle(frame, (center_x, center_y), 5, color, -1)

    # Write the frame to the video file
    out.write(frame)

    # Display the frame
    cv2.imshow('YOLO Human Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Close the video file
cv2.destroyAllWindows()
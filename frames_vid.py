import cv2
import os
import glob

# Folder containing frames
input_folder = "/home/gazebo/Downloads/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/image_02/data"

# Output video
output_video = "/home/gazebo/src/YOLO-3D/input/kiit_ 30FPS.mp4"

# FPS (frames per second)
fps = 30

# Get all image files
images = sorted(glob.glob(os.path.join(input_folder, "*.png")))

if not images:
    print("No images found!")
    exit()

# Read first frame to get width and height
first_frame = cv2.imread(images[0])
height, width = first_frame.shape[:2]

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Add frames to video
for image in images:
    frame = cv2.imread(image)

    if frame is not None:
        video.write(frame)

# Release video
video.release()

print(f"Video saved as: {output_video}")
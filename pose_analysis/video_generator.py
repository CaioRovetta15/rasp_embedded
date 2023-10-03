import cv2
import numpy as np

# Set the parameters
output_video_filename = "output_video.mp4"
image_filename = "talk.png"  # Replace with your image file
frame_duration = 7  # 7 seconds
frame_rate = 15  # 15 frames per second

# Load the image
image = cv2.imread(image_filename)
if image is None:
    raise Exception(f"Error: Could not load image from {image_filename}")

# Get the image dimensions
height, width, layers = image.shape

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
fps = frame_rate
out = cv2.VideoWriter(output_video_filename, fourcc, fps, (width, height))

# Write the image to the video for the specified duration and frame rate
frame_count = int(frame_duration * fps)
for _ in range(frame_count):
    out.write(image)

# Release the VideoWriter
out.release()

print(f"Video saved as {output_video_filename}")

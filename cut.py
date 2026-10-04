from moviepy import VideoFileClip
import json

# Input video
input_video = "/home/gazebo/src/YOLO-3D/output/output_metric_venice_lbo.mp4"

# Output video
output_video = "/home/gazebo/src/YOLO-3D/output/output_metric_venice_lbo_trimed.mp4"

# Start and end timestamps in seconds
start_time = 0
end_time = 10

# Load video
video = VideoFileClip(input_video)

# Trim video (original video is NOT modified)
trimmed = video.subclipped(start_time, end_time)

# Save trimmed video
trimmed.write_videofile(output_video, codec="libx264", audio_codec="aac")

# Save timestamps
timestamps = {
    "original_video": input_video,
    "start_time_seconds": start_time,
    "end_time_seconds": end_time,
    "duration_seconds": end_time - start_time
}

with open("timestamps.json", "w") as f:
    json.dump(timestamps, f, indent=4)

# Close files
trimmed.close()
video.close()

print("Done!")
print(f"Trimmed video saved as: {output_video}")
print("Original video was not modified.")
print("Timestamps saved as: timestamps.json")
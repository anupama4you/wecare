# Step 1: This File converts a video to a set of images
from pathlib import Path
import cv2
import os

def Video2ImageConvertor(video_path:Path, output_folder:Path, desired_frame_count:int):
    """
    Extracts frames from a video and saves them as separate images

    Params:
        video_path: Path to the video file
        output_folder: Path to the folder where the images will be saved
        desired_frame_count: Number of frames to be extracted from the video
    
    Returns:
        None, but saves the extracted frames to the output folder
    """
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Couldn't open the video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_interval = total_frames / desired_frame_count

    frame_number = 0
    extracted_count = 0
    while extracted_count < desired_frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Jump to the specified frame number
        ret, frame = cap.read()

        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"{extracted_count}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_number = int(frame_number + skip_interval)
        extracted_count += 1

    cap.release()
    print(f"[INFO] Extracted {extracted_count} frames and saved them to {output_folder}")
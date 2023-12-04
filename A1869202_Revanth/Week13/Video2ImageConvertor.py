# This is replacement of PreprocessingData.py

from pathlib import Path
import argparse
import cv2
import os


def Video2ImageConvertor(video_path: Path, output_folder: Path, desired_frame_count: int):
    """
    Extracts frames from a video and saves them as separate images

    Params:
        video_path: Path to the video file (individual patient video)
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
        # Jump to the specified frame number
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"{extracted_count}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_number = int(frame_number + skip_interval)
        extracted_count += 1

    cap.release()

    print(
        f"[INFO] Extracted {extracted_count} frames and saved them to {output_folder}")
def delete_blobs_in_folder(bucket_name, folder_prefix):
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_prefix)  # list blobs in the folder
    for blob in blobs:
        blob.delete()

    print(f"All blobs in {folder_prefix} have been deleted.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--patients_videos_path",
                    required=True, help="Path to the video file")
    ap.add_argument("-o", "--patients_images_path", required=False,
                    help="Path to the folder where the images will be saved", default="Patients_Images")
    ap.add_argument("-fc", "--desired_frame_count", required=False,
                    help="Number of frames to be extracted from the video", default=25)
    args = vars(ap.parse_args())

    patients_videos_path = Path(args["patients_videos_path"])
    patients_images_path = Path(args["patients_images_path"])

    # Check if the given videoPath exists
    for patient in os.listdir(args["patients_videos_path"]):
        # only pass video files like - .mp4, .avi, .mov
        if not patient.endswith(".mp4") and not patient.endswith(".avi") and not patient.endswith(".mov"):
            continue

        # Only print the patient name, not the extension
        print("[INFO] Processing: Patient ID: ", os.path.splitext(patient)[0])
        cur_patient_video_path = os.path.join(patients_videos_path, patient)
        cur_patient_image_path = os.path.join(
            patients_images_path, os.path.splitext(patient)[0])

        Video2ImageConvertor(cur_patient_video_path, cur_patient_image_path,
                             desired_frame_count=int(args["desired_frame_count"]))

    print("[INFO] Processing done!")

# Sample Command:
# python Video2ImageConvertor.py -v Patients_Videos -o Patients_Images -fc 20

# But run with default values like this:
# python Video2ImageConvertor.py -v Patients_Videos

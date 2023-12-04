from google.cloud import storage
from pathlib import Path
from imutils import paths
import argparse
import cv2
import os


bucket_name = "watchdog-gamma.appspot.com"
storage_client = storage.Client()


def PreprocessingData(roomNum: str, filePath: str):
    bucket = storage_client.get_bucket(bucket_name)
    desired_frame_count = 25
    blob = bucket.blob(filePath)
    parts = filePath.split("/")
    file_name = parts[2]
    local_video_path = f"/tmp/{file_name}"
    cur_patient_image_prefix = f"/tmp/images"
    blob.download_to_filename(local_video_path,)
    delete_blobs_in_folder(bucket_name, f'Patients_Images/{roomNum}/')
    # Apply video to image converter
    Video2ImageConvertor(
        local_video_path, cur_patient_image_prefix, desired_frame_count,roomNum)
    
    if os.path.exists(local_video_path):
        os.remove(local_video_path)


def Video2ImageConvertor(video_path: Path, output_folder: Path, desired_frame_count: int, roomNum: int):
    """
    Extracts frames from a video and saves them as separate images

    Params:
        video_path: Path to the video file (individual patient video)
        output_folder: Path to the folder where the images will be saved
        desired_frame_count: Number of frames to be extracted from the video

    Returns:
        None, but saves the extracted frames to the output folder
    """
    imagePaths = list(paths.list_images(output_folder))
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
        upload_blob(bucket_name, frame_filename,
                    f'Patients_Images/{roomNum}/{Path(frame_filename).name}')
        frame_number = int(frame_number + skip_interval)
        extracted_count += 1

    cap.release()
    print(
        f"[INFO] Extracted {extracted_count} frames and saved them to {output_folder}")


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def delete_blobs_in_folder(bucket_name, folder_prefix):
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_prefix)  # list blobs in the folder
    for blob in blobs:
        blob.delete()

    print(f"All blobs in {folder_prefix} have been deleted.")


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-v", "--patients_videos_path",
#                     required=True, help="Path to the video file")
#     ap.add_argument("-o", "--patients_images_path", required=False,
#                     help="Path to the folder where the images will be saved", default="Patients_Images")
#     ap.add_argument("-fc", "--desired_frame_count", required=False,
#                     help="Number of frames to be extracted from the video", default=25)
#     args = vars(ap.parse_args())

#     patients_videos_path = Path(args["patients_videos_path"])
#     patients_images_path = Path(args["patients_images_path"])

#     # Check if the given videoPath exists
#     for patient in os.listdir(args["patients_videos_path"]):
#         # only pass video files like - .mp4, .avi, .mov
#         if not patient.endswith(".mp4") and not patient.endswith(".avi") and not patient.endswith(".mov"):
#             continue

#         # Only print the patient name, not the extension
#         print("[INFO] Processing: Patient ID: ", os.path.splitext(patient)[0])
#         cur_patient_video_path = os.path.join(patients_videos_path, patient)
#         cur_patient_image_path = os.path.join(
#             patients_images_path, os.path.splitext(patient)[0])

#         Video2ImageConvertor(cur_patient_video_path, cur_patient_image_path,
#                              desired_frame_count=int(args["desired_frame_count"]))

#     print("[INFO] Processing done!")

# Sample Command:
# python Video2ImageConvertor.py -v Patients_Videos -o Patients_Images -fc 20

# But run with default values like this:
# python Video2ImageConvertor.py -v Patients_Videos

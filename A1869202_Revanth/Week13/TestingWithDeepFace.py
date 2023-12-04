import cv2
import mediapipe as mp
import tensorflow as tf
from deepface import DeepFace
import pandas as pd
import base64

# Constants
target_size = (96, 96, 3)
room_no = "0"

# Model Related constants
backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'fastmtcnn',
]

models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]

min_distance_threshold = 0.7
min_identites_threshold = 1

metrics = ["cosine", "euclidean", "euclidean_l2"] # Euclidean L2 form seems to be more stable than cosine and regular Euclidean distance based on experiments.

# Create a VideoCapture object to read from camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    results = DeepFace.find(img_path=frame, db_path="Patients_Images", model_name=models[2], distance_metric=metrics[2], enforce_detection=False, detector_backend=backends[5])
    #print(results)

    # Returned df has following colomns = [identity  source_x  source_y  source_w  source_h  Facenet_euclidean_l2]
    # Get matches count of the Face with given Room Number (Returned path will be like: Patients_Images/RoomNo/1.jpg -> Patients_Images/0/8.jpg)
    matches_count, minimum_distance = 0, float('inf')

    for data_frame in results:
        actual_matches_found = data_frame.shape[0]
        for index, row in data_frame.iterrows():
            if (row['identity'].split("/")[-2] == room_no) and (row['Facenet512_euclidean_l2'] <= min_distance_threshold):
                matches_count += 1
                minimum_distance = min(minimum_distance, row['Facenet512_euclidean_l2'])
    print(f"As per Conditions: Matches: {matches_count}\{actual_matches_found}, Minimum Distance: {minimum_distance}")
    
    # If matches count is greater than threshold, then the person is identified
    if matches_count >= min_identites_threshold:
        # set the text in frame to "Room No: room_no", with green color and thickness 2 & font 1, set Count of matches and minimum distance in frame
        cv2.putText(frame, "Room No: " + str(room_no), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Matches: {matches_count}\{actual_matches_found}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Minimum Distance: {minimum_distance}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Frames', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()

import base64
import time
import argparse
from threading import Thread
from collections import deque
import firebase_admin
from firebase_admin import credentials, db
import tensorflow as tf
from deepface import DeepFace

# ----------------------------------------------------------------------------------------------- #
# Firebase configuration
cred = credentials.Certificate('watchdog-gamma-firebase-adminsdk-nbg3v-135a49e821.json')  # Replace with the path to your Firebase service account key
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://roomexitdetection.asia-southeast1.firebasedatabase.app/'  # Replace with your Realtime Database URL
})

# Get a database reference
database = db.reference()

# ----------------------------------------------------------------------------------------------- #
# Thread 1: Data Extraction Thread
shared_data_queue = deque()  # Using deque for thread-safe operations

def data_extraction_thread():
    try:
        print("[INFO] T1: Starting data extraction thread..")
        
        detected_faces_db = database.child('DetectedFaces')
        last_processed_id = None
        while True:
            if last_processed_id:
                new_data = detected_faces_db.order_by_key().start_at(last_processed_id).get()
            else:
                new_data = detected_faces_db.get()

            if (new_data and (len(new_data) > 1 or last_processed_id is None)):
                for key, entry in new_data.items():
                    shared_data_queue.append((entry["face"], entry["distance"], entry["room_no"]))
                    last_processed_id = key

            # if there are no items in new_data, then sleep for 1 second
            if not new_data or len(new_data) <= 1:
                time.sleep(1)

    except KeyboardInterrupt:
        print("[INFO] T1: Exiting data extraction thread (KeyboardInterrupt or unexpected error)..")

# ----------------------------------------------------------------------------------------------- #
# Thread 2: Data Processing Thread
processed_data_map = {}

# Model Related constants
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet', 'fastmtcnn',]
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace",]
metrics = ["cosine", "euclidean", "euclidean_l2"]
min_distance_threshold = 0.8
min_identites_threshold = 1

def data_processing_thread(Patient_images_folder: str = "Patients_Images"):
    try:
        # Check if "ROOMEXITS", if not then create it
        room_exit_db = database.child("ROOMEXITS")

        print("[INFO] T2: Starting data processing & RoomExit Monitoring thread..")
        
        while True:
            if shared_data_queue.__len__()>0:
                face, distance, room_no = shared_data_queue.popleft()

                # Whatever face we get is base64 encoded, so we need to decode it first
                face = base64.b64decode(face)
                face = tf.io.decode_image(face, channels=3, dtype=tf.dtypes.uint8)
                face = tf.image.resize(face, (224, 224))
                face = tf.keras.preprocessing.image.img_to_array(face)

                results = DeepFace.find(img_path=face, db_path=f"{Patient_images_folder}/{room_no}", model_name=models[2], distance_metric=metrics[2], enforce_detection=False, detector_backend=backends[5])
                try:
                    matches_count = results[0].shape[0]
                    minimum_distance = results[0]['Facenet512_euclidean_l2'][0] # Returned results are already sorted based on distance (ascending order)
                except:
                    matches_count = 0
                    minimum_distance = float('inf')           
                actual_matches = matches_count

                if minimum_distance > min_distance_threshold:
                    matches_count = 0
                         
                print(f"[INFO] Room ID: {room_no}' | Found at a distance: {distance} | Matches: {matches_count}/{actual_matches} | Minimum Distance: {minimum_distance}.")
                
                # Update processed_data_map: if matched_probability > 35%, means this is the patient in the given room
                if matches_count >= min_identites_threshold:
                    if room_no in processed_data_map:
                        processed_data_map[room_no]["prev_prev_distance"] = processed_data_map[room_no]["prev_distance"]
                        processed_data_map[room_no]["prev_distance"] = distance
                    else:
                        processed_data_map[room_no] = {"prev_distance": distance, "prev_prev_distance": distance, "room_no": room_no, "room_exitted": None}
                    
                    # Check if the room_no is in local_alarm_map or not, if not then check whether its exit or not, if room_no already exists, then check when is last notification sent time, if more than 1 minute, send notification again
                    patient_exitted_room = (processed_data_map[room_no]["prev_distance"]<processed_data_map[room_no]["prev_prev_distance"])

                    # In processed data Map store, the last time status of person, Event based notification (If event changes, then only send notification)
                    if ((processed_data_map[room_no]["room_exitted"] is None) or (processed_data_map[room_no]["room_exitted"]!=patient_exitted_room)):
                        # Send update to Firebase
                        room_exit_db.child(str(room_no)).set({"Room_Exited": patient_exitted_room, "Prev_distance": processed_data_map[room_no]["prev_distance"], "Prev_Prev_distance": processed_data_map[room_no]["prev_prev_distance"]})
                        processed_data_map[room_no]["room_exitted"] = patient_exitted_room
            else:
                time.sleep(1)  

    except KeyboardInterrupt:
        print("[INFO] T2: Exiting data processing & RoomExit Monitoring thread (KeyboardInterrupt or unexpected error)..")

# ----------------------------------------------------------------------------------------------- #
# Start all the threads
if __name__ == "__main__":
    
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--Patient_images_folder", required=False,
        help="Path to the patients images folder", default="Patients_Images")
    args = vars(ap.parse_args())

    t1 = Thread(target=data_extraction_thread)
    t2 = Thread(target=data_processing_thread, args=(args["Patient_images_folder"],))
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()

# ----------------------------------------------------------------------------------------------- #

# To run this script, use the following command:
# python Faces_2_RoomExit_V3.py

# Updates Made: (For Kaifeng)
# 1. In thread 2 the code is modified to use, DeepFace.
# 2. Argsparse is changed in main. 
# So update Thread 2 related code and main function

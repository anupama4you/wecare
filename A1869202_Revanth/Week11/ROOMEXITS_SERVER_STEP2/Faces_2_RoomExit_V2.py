import base64
import time
import argparse
from threading import Thread
from collections import deque
import firebase_admin
from firebase_admin import credentials, db
import tensorflow as tf
import Predictions_tflite as predictor

# ----------------------------------------------------------------------------------------------- #
# Firebase configuration
cred = credentials.Certificate('watchdog-gamma-firebase-adminsdk-nbg3v-135a49e821.json')  # Replace with the path to your Firebase service account key
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://roomexits.firebaseio.com/'  # Replace with your Realtime Database URL
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
def data_processing_thread(model_name: str = "MOBILENETV2", target_size:tuple = (96, 96, 3), Patient_images_folder: str = "Patients_Images"):
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
                face = tf.image.resize(face, target_size[:2])
                face = tf.reshape(face, target_size)

                # Perform face recognition and other processing (using predictor.Predictions())
                matched_probability = float(predictor.Predictions(room_no, face, model_name, target_size, Patient_images_folder))
                print(f"[INFO] Room ID: {room_no}'patient is found on doorway, at a distance of {distance} matched with {matched_probability}%")
                
                # Update processed_data_map: if matched_probability > 35%, means this is the patient in the given room
                if matched_probability>50.0:
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
    ap.add_argument("-m", "--model_name", required=False,
        help="Path to the trained model", default="MOBILENETV2")
    ap.add_argument("-s", "--target_size", required=False,
        help="desired height/width shape of the image, For Model: Custom: 64 -> (64, 64, 3), MobileNetV2: 224 -> (224, 224, 3), for others: 300 -> (300, 300, 3)", default=96)
    args = vars(ap.parse_args())

    t1 = Thread(target=data_extraction_thread)
    t2 = Thread(target=data_processing_thread, args=(args["model_name"], (int(args["target_size"]), int(args["target_size"]), 3), args["Patient_images_folder"]))
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()

# ----------------------------------------------------------------------------------------------- #

# To run this script, use the following command:
# python Faces_2_RoomExit_V2.py

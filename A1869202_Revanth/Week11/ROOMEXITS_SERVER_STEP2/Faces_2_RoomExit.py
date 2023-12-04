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

            if not new_data or len(new_data) <= 1:
                time.sleep(1)

    except KeyboardInterrupt:
        print("[INFO] T1: Exiting data extraction thread (KeyboardInterrupt or unexpected error)..")


# ----------------------------------------------------------------------------------------------- #
# Thread 2: Data Processing Thread
processed_data_map = {}
def data_processing_thread(model_name: str = "MOBILENETV2", target_size:tuple = (96, 96, 3), Patient_images_folder: str = "Patients_Images"):
    try:
        print("[INFO] T2: Starting data processing thread..")

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
                        processed_data_map[room_no]["timestamp_processed"] = time.time() # last time it was processed this room's data, when patients face was matched/recognized.
                        processed_data_map[room_no]["prev_prev_distance"] = processed_data_map[room_no]["prev_distance"]
                        processed_data_map[room_no]["prev_distance"] = distance
                    else:
                        processed_data_map[room_no] = {"timestamp_processed": time.time(), "prev_distance": distance, "prev_prev_distance": distance, "room_no": room_no}
            else:
                time.sleep(1)  

    except KeyboardInterrupt:
        print("[INFO] T2: Exiting data processing thread (KeyboardInterrupt or unexpected error)..")

# ----------------------------------------------------------------------------------------------- #
# Thread 3: Monitoring Thread
local_alarm_map = {}
def monitoring_thread(send_notification_after_every: float = 60000.0):
    try:
        print("[INFO] T3: Starting monitoring thread..")
        # Check if "ROOMEXITS", if not then create it
        room_exit_db = database.child("ROOMEXITS")
        
        while True:
            # give time in milliseconds
            current_time = time.time()
            # as per time.time() gives time in milliseconds, so 60 seconds = 60*1000 milliseconds
            for room_no, data in processed_data_map.items():
                # Check if the room_no is in local_alarm_map or not, if not then check whether its exit or not, if room_no already exists, then check when is last notification sent time, if more than 1 minute, send notification again
                did_person_returned_to_room = (data["prev_distance"]>=data["prev_prev_distance"])
                if room_no not in local_alarm_map or (did_person_returned_to_room) or ((local_alarm_map[room_no] - data["timestamp_processed"])>=send_notification_after_every):
                    # Send update to Firebase
                    if did_person_returned_to_room:
                        room_exit_db.child(str(data["room_no"])).set({"Room_Exited": False, "Prev_distance": data["prev_distance"], "Prev_Prev_distance": data["prev_prev_distance"]})
                    else:
                        room_exit_db.child(str(data["room_no"])).set({"Room_Exited": True, "Prev_distance": data["prev_distance"], "Prev_Prev_distance": data["prev_prev_distance"]})
                        local_alarm_map[room_no] = current_time
            
            # if there are no items in processed_data_map, then sleep for 1 second
            if len(processed_data_map)==0:
                time.sleep(1)

    except KeyboardInterrupt:
        print("[INFO] T3: Exiting monitoring thread (KeyboardInterrupt or unexpected error)..")

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
    ap.add_argument("-d", "--notification_delay", required=False, 
                    help="The delay after which the notification should be sent to the user, in milliseconds", default=60000.0)
    args = vars(ap.parse_args())

    t1 = Thread(target=data_extraction_thread)
    t2 = Thread(target=data_processing_thread, args=(args["model_name"], (int(args["target_size"]), int(args["target_size"]), 3), args["Patient_images_folder"]))
    t3 = Thread(target=monitoring_thread, args=(args["notification_delay"],))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

# ----------------------------------------------------------------------------------------------- #

# To run this script, use the following command:
# python Faces_2_RoomExit.py

import pyrebase

config = {
    "apiKey": "AIzaSyBMd8nI8zniWNtQ5RvVRitmpgUpV1ucUYk",
    "authDomain": "watchdog-gamma.firebaseapp.com",
    "databaseURL": "https://roomexits.firebaseio.com/",
    "projectId": "watchdog-gamma",
    "storageBucket": "watchdog-gamma.appspot.com",
    "messagingSenderId": "503315913339",
    "appId": "1:503315913339:web:7a0951b5c54c424c7420de",
    "measurementId": "G-P99HVG61KT"
}
firebase = pyrebase.initialize_app(config)
database = firebase.database()

# database.child("DetectedFaces").child(FrameID).push({"FaceEncoding": face_base64, "Distance": distance}) -> to push data to firebase
# database.child("DetectedFaces").child(FrameID).get().val() -> to get data from firebase

def get_data():
    data = database.child("DetectedFaces").child(FrameID).get().val()
    return data
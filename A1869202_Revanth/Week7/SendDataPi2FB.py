import pyrebase
import random

config = {
  "apiKey": "AIzaSyAvtzWT-oMPYbR6elmiTVQXKCmtKfoZZ-w",
  "authDomain": "facerecognition-41558.firebaseapp.com",
  "databaseURL": "https://facerecognition-41558-default-rtdb.firebaseio.com",
  "projectId": "facerecognition-41558",
  "storageBucket": "facerecognition-41558.appspot.com",
  "messagingSenderId": "1011726147408",
  "appId": "1:1011726147408:web:b849bc2dc9cd8041291655",
  "measurementId": "G-30PFLV1MRS"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
database = firebase.database()

for i in range(4):
  for j in range(6):
    randomList = [round(random.uniform(1.0, 10.0), 2) for _ in range(128)]
    data = {
      "FaceEmbedding" : randomList,
      "Distance": random.randint(1, 100)
    }
    database.child("FaceData").child(i).push(data)
    print("data sent!")


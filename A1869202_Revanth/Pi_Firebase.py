# sudo apt-get install python3-pip
# Installing firebase command for Pi => sudo pip3 install pyrebase

import time
import pyrebase
import random

config = {
  "apiKey": "x",
  "authDomain": "connectpi2firebase-1.firebaseapp.com",
  "databaseURL": "https://connectpi2firebase-1-default-rtdb.firebaseio.com",
  "storageBucket": "connectpi2firebase-1.appspot.com"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()
while True:
  num1 = random.randint(0, 100)
  num2 = random.randint(0, 100)
  print("Random numbers generated are: ", num1, num2)
  data = {
    "Number_1": num1,
    "Number_2": num2
  }
  db.child("status").push(data)
  print("Data is sucessfully sent to pyrebase")
  time.sleep(2)
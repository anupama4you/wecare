# pip3 install mediapipe
# Cmd: python3 FaceDetectionVideo.py

import cv2
import mediapipe as mp
import time
import imutils
import pickle
import os
import math
from imutils.video import FPS
import pyrebase
import random
import base64
import multiprocessing

terminate_flag = multiprocessing.Value('i', 0) 

class FaceDetector():
    def __init__(self, modelSelection=1, minDetectionCon=0.5):

        self.minDetectionCon = minDetectionCon
        self.modelSelection = modelSelection

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.modelSelection, self.minDetectionCon)
        
    def findFaces(self, img, draw=True):
        
        # To find distance of person from camera
        F = 855 
        W = 8.9 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Calculating the distance between 2 ending points in pixels (middle of box's - left height line, right height line)
                w = math.dist([bbox[0], bbox[1]+(bbox[3]/2)], [bbox[0] + bbox[2], bbox[1]+(bbox[3]/2)])

                # Calculating the Distance (D) of the person from the camera
                D = (W * F) / w

                bboxs.append([id, bbox, detection.score, D])
            
        return img, bboxs

def face_detection_process(shared_img_buffer, database, EXIT_RANGE, terminate_flag):
    # Initialising FaceDetection Object
    detector = FaceDetector()
    print("[INFO] Process 1: Started Detecting faces.")
    fps1 = FPS().start()
    FrameID = 0

    while True:
        if terminate_flag.value == 1:
            break

        img = shared_img_buffer[0]
        if img is None:
            continue

        img, bboxs = detector.findFaces(img, False)
        FrameID += 1

        if len(bboxs) != 0:
            try:
                for i in range(len(bboxs)):
                    (startX, startY, w, h) = bboxs[i][1]
                    distance = bboxs[i][3]

                    if distance > EXIT_RANGE:
                        continue

                    face = img[startY:startY+h, startX:startX+w]
                    _, buffer = cv2.imencode('.jpg', face)
                    face_base64 = base64.b64encode(buffer).decode('utf-8')
                    database.child("DetectedFaces").child(FrameID).push({"FaceEncoding": face_base64, "Distance": distance})
            except:
                pass

        fps1.update()

    fps1.stop()
    print("[INFO] elasped time: {:.2f}".format(fps1.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps1.fps()))
    print("[INFO] Process 1: Done encoding faces. Exiting Process ....")


def fall_detection_process(shared_img_buffer, config, terminate_flag):
    while True:
        if terminate_flag.value == 1:
            break
            
        img = shared_img_buffer[0]
        if img is None:
            continue

        # Fall detection code goes here -- will be fixed this week


def main():
    EXIT_RANGE = 140
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
    database = firebase.database()

    manager = multiprocessing.Manager()
    shared_img_buffer = manager.list()
    shared_img_buffer.append(None)

    face_process = multiprocessing.Process(target=face_detection_process, args=(shared_img_buffer, database, EXIT_RANGE, terminate_flag))
    fall_process = multiprocessing.Process(target=fall_detection_process, args=(shared_img_buffer, database, terminate_flag))

    face_process.start()
    fall_process.start()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            continue
        shared_img_buffer[0] = img
        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    terminate_flag.value = 1
    face_process.join()
    fall_process.join()

if __name__ == "__main__":
    main()

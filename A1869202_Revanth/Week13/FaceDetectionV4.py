# pip3 install mediapipe
# pip3 install pyrebase
# Cmd: python3 FaceDetectionVideo.py

import cv2
import mediapipe as mp
import time
import imutils
import os
import math
from imutils.video import FPS
import pyrebase
import base64
import argparse
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
        F = 855 # Calculated based on experimenting with the camera
        W = 8.9 # Width of the face in centimeters (Average width of a human face)
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

def face_detection_process(shared_img_buffer, database, EXIT_RANGE, ROOM_NO, terminate_flag):
    frame_height = 360
    
    try:
        # Initialising FaceDetection Object
        detector = FaceDetector()
        print("[INFO] Process 1: Started Detecting faces.")
        fps1 = FPS().start()
        
        while True:
            if terminate_flag.value == 1:
                break

            img = shared_img_buffer[0]
            if img is None:
                continue

            img, bboxs = detector.findFaces(img, False)
            
            if len(bboxs) != 0:
                try:
                    for i in range(len(bboxs)):
                        (startX, startY, w, h) = bboxs[i][1]
                        distance = bboxs[i][3]
                        
                        if distance > EXIT_RANGE:
                            continue

                        face = img[startY:startY+h, startX:startX+w]

                        cv2.rectangle(img, (startX, startY), (startX+w, startY+h), (0, 255, 0), 2)
                        cv2.imshow("Image", img)
                        cv2.waitKey(1)

                        # Updated with "from how much height the face is detected in the frame, higher the value, closer the person is to the camera, 
                        # lower the value, farther the person is to the camera. So if higher value, means person is exiting room. 
                        # Because based on the way we placed the camera in the room, as the person leaves, the face goes from top to bottom of the frame."
                        distance = (frame_height - bboxs[i][1][1])
                        _, buffer = cv2.imencode('.jpg', face)
                        face_base64 = base64.b64encode(buffer).decode('utf-8')
                        database.child("DetectedFaces").push({"face": face_base64, "distance": distance, "room_no": ROOM_NO})
                except:
                    pass

            fps1.update()

        fps1.stop()
        print("[INFO] elasped time: {:.2f}".format(fps1.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps1.fps()))
        print("[INFO] Process 1: Done encoding faces. Exiting Process ....")
    
    except KeyboardInterrupt:
        fps1.stop()
        print("[INFO] elasped time: {:.2f}".format(fps1.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps1.fps()))
        print("[INFO] Process 1: Interrupted the child process. Exiting Process ....")

def main(ROOM_NO: int=1, EXIT_RANGE: int = 180):
    config = {
        "apiKey": "AIzaSyBMd8nI8zniWNtQ5RvVRitmpgUpV1ucUYk",
        "authDomain": "watchdog-gamma.firebaseapp.com",
        "databaseURL": "https://roomexitdetection.asia-southeast1.firebasedatabase.app",
        "projectId": "watchdog-gamma",
        "storageBucket": "watchdog-gamma.appspot.com",
        "messagingSenderId": "503315913339",
        "appId": "1:503315913339:web:7a0951b5c54c424c7420de",
        "measurementId": "G-P99HVG61KT"
    }
    firebase = pyrebase.initialize_app(config)
    database = firebase.database()
    database.child("DetectedFaces").remove()
    database.child("ROOMEXITS").remove()

    manager = multiprocessing.Manager()
    shared_img_buffer = manager.list()
    shared_img_buffer.append(None)

    face_process = multiprocessing.Process(target=face_detection_process, args=(shared_img_buffer, database, EXIT_RANGE, ROOM_NO, terminate_flag))
    
    face_process.start()
    
    try:
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                continue
            shared_img_buffer[0] = img
            """
            cv2.imshow("img", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            """
    except KeyboardInterrupt:
        print("[INFO] Terminating the process ...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

        terminate_flag.value = 1
        face_process.join()
    
if __name__ == "__main__":
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--distance", type=int, default=180, help="Minimum Distance of person from camera to start detecting face. MAX ALLOWED < 300 CENTIMETERS", required=False)
    ap.add_argument("-r", "--roomno", type=int, default=1, help="Enter the Room Number, where this pi is installed", required=False)
    args = vars(ap.parse_args())

    EXIT_RANGE = args["distance"]
    roomNo = args["roomno"]

    main(roomNo, EXIT_RANGE)

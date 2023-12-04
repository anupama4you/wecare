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

class FaceDetector():
    def __init__(self, modelSelection=1, minDetectionCon=0.5):

        self.minDetectionCon = minDetectionCon
        self.modelSelection = modelSelection

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.modelSelection, self.minDetectionCon)
        
    def findFaces(self, img, draw=True):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)

                bboxs.append([id, bbox, detection.score])
            
        return img, bboxs


def main():
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

    cap = cv2.VideoCapture(0)
    pTime = 0
    
    detector = FaceDetector()
    time.sleep(3.0)
    print("[INFO] Reading faces. Press 'q' to stop.")
    fps1 = FPS().start()
    FrameID = 0
    print("[INFO] Detecting faces... ")
    while cap.isOpened():
        success, img = cap.read()
        
        img, bboxs = detector.findFaces(img, False)
        FrameID += 1
        
        if (len(bboxs)!=0):
            try:
                for i in range(len(bboxs)):
                    # compute the facial embedding for the face                
                    (startX, startY, w, h) = bboxs[i][1]
                    endX = startX+w
                    endY = startY+h
                    
                    # extract the face ROI and grab the ROI dimensions: Ensure startX is smaller than endX and startY is smaller than endY
                    face = img[startY:endY, startX:endX]
                    cv2.imshow("Face", face)
                    
                    # Convert the face ROI to a base64 string
                    retval, buffer = cv2.imencode('.jpg', face)
                    face_base64 = base64.b64encode(buffer).decode('utf-8')
                    data = {
                        "Face": face_base64,
                    }
                    database.child("DetectedFaces").child(FrameID).push(data)

                print("[INFO] Done encoding faces. ")

            except:
                pass  # No face found in image

        # update the FPS counter
        fps1.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps1.stop()
    print("[INFO] elasped time: {:.2f}".format(fps1.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps1.fps()))
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
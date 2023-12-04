# pip3 install mediapipe
# Cmd: python buildFaceEncodings.py --dataset collected_faces --encodings faceEncodings.pickle

import cv2
import mediapipe as mp
import time
import imutils
import argparse
import os
import sys
import pickle
import face_recognition
from imutils import paths

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        # returns faces - bounding boxes
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
        return bboxs

def print_progress_bar(name, iteration, total, bar_length=50):
    if iteration == 1:
        return  # print a newline to ensure we start on a new line
    progress = (iteration / total)
    arrow = '-' * int(round(progress * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    name = name + "'s face:"
    sys.stdout.write(f'\r[INFO] Encoding: {name} {arrow + spaces} {iteration}/{total}')
    sys.stdout.flush()

def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", required=True,
        help="path to input directory of faces + images")
    ap.add_argument("-e", "--encodings", required=True,
        help="path to serialized db of facial encodings")
    args = vars(ap.parse_args())

    # grab the paths to the input images in our dataset
    imagePaths = list(paths.list_images(args["dataset"]))

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []

    # load the FaceDetector class
    detector = FaceDetector()
    didntFindFaces = 0
    totalImages = len(imagePaths)

    # loop over the image paths 
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        name = imagePath.split(os.path.sep)[-2]
        print_progress_bar(name, i + 1, len(imagePaths))
        
        # load the input image and detect faces in it and find bounding boxes
        img = cv2.imread(imagePath)
        bboxs = detector.findFaces(img)
        # print("x:", bboxs[0][0], "1:", bboxs[0][1], "2: ", bboxs[0][2], "y")

        # compute the facial embedding for the face
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encodings = face_recognition.face_encodings(rgb, [bboxs[0][1]])
            # print("encodings:", len(encodings[0]))

            # loop over the encodings
            for encoding in encodings:
                # add each encoding + name to our set of known names and encodings
                knownEncodings.append(encoding)
                print("Name:", name)
                knownNames.append(name)
                # size_of_single_encoding = sys.getsizeof(encoding)
                # print(f"Size of a single encoding: {size_of_single_encoding} bytes")
        except:
            didntFindFaces += 1 # "No face found in image"

    # dump the facial encodings + names to disk
    # print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(args["encodings"], "wb")
    f.write(pickle.dumps(data))
    f.close()

    print(f"\n[INFO] Model has an accuracy of: {((totalImages-didntFindFaces)/totalImages)*100}%, found faces in {(totalImages-didntFindFaces)} images, out of {totalImages}.")
    print("[INFO] Processing completed.")

if __name__ == "__main__":
    main()
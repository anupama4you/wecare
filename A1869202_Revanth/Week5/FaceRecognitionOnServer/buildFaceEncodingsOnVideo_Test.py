# pip3 install mediapipe
# Cmd: python3 FaceDetectionVideo.py

import cv2
import mediapipe as mp
import time
import imutils
import pickle
import os
import face_recognition
from imutils.video import FPS

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def resize_image(img, scale_factor=1.5):
        """
        Resize the image by a given scale factor.
        
        Parameters:
        - img: Input image.
        - scale_factor: Factor by which to resize the image.
        
        Returns:
        Resized image.
        """
        height, width, other = img.shape
        new_width, new_height = int(width * scale_factor), int(height * scale_factor)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img
    
    def zoom_image(self, img, zoom_factor):
        height, width = img.shape[:2]
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
        
        # determine the delta width and height to trim off an image
        delta_w = new_width - width
        delta_h = new_height - height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # resize the image with the computed dimensions
        img_zoomed = cv2.resize(img, (new_width, new_height))

        # crop the image back to the original size
        img_zoomed = img_zoomed[top:img_zoomed.shape[0]-bottom, left:img_zoomed.shape[1]-right]
        
        return img_zoomed

    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img,bbox)

                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 0, 255), 2)
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5, rt= 1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        # Rectangle
        cv2.rectangle(img, bbox, (0, 0, 255), rt)
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), (0, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (0, 0, 255), t)
        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (0, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (0, 0, 255), t)
        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + l, y1), (0, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0, 0, 255), t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (0, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (0, 0, 255), t)

        return img


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    foundEncodings = [] # initialize the list of found encodings
    detector = FaceDetector()
    filename = "facesFoundInVideo_Encodings.pickle"
    time.sleep(3.0)
    print("[INFO] Reading faces. Press 'q' to stop.")
    fps1 = FPS().start()
    alternative = False
    while True:
        success, img = cap.read()
        print("Image shape: ", img.shape)
        print("[INFO] Detecting faces... ")

        # Resize (zoom into) the image
        #if alternative:
        img = detector.zoom_image(img, zoom_factor=1.4)
        #alternative ^= True

        img, bboxs = detector.findFaces(img)
        # print(bboxs) # [[0, (608, 258, 245, 245), [0.9235379695892334]]]

        try:
            print("[INFO] Encoding faces... ")
            # compute the facial embedding for the face
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb, [bboxs[0][1]])
            
            # loop over the encodings (Here we will pass the encodings to the server, but for now we will just store them in a list)
            for encoding in encodings:
                foundEncodings.append(encoding)
            
            # Check if the file exists and load its data if it does
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    old_data = pickle.load(f)
                    foundEncodings.extend(old_data["encodings"])

            # Only keep the last 20 encodings
            final_encodings = foundEncodings[-20:]

            # Update or create the file with the appended data
            data = {"encodings": final_encodings}
            with open(filename, "wb") as f:
                f.write(pickle.dumps(data))

        except:
            pass  # No face found in image
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        # update the FPS counter
        fps1.update()
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps1.stop()
    print("[INFO] elasped time: {:.2f}".format(fps1.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps1.fps()))
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
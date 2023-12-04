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

class FaceDetector():
    def __init__(self, modelSelection=1, minDetectionCon=0.5):

        self.minDetectionCon = minDetectionCon
        self.modelSelection = modelSelection

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.modelSelection, self.minDetectionCon)

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
        
        # To find distance of person from camera
        F = 840 # Focal length in Centimeters
        W = 6.3 # Distance b/w eyes in real life (avg of men and women)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # For each face found find distance b.w eyes
                left_eye_x = detection.location_data.relative_keypoints[0].x * img.shape[1]
                right_eye_x = detection.location_data.relative_keypoints[1].x * img.shape[1]
                left_eye_y = detection.location_data.relative_keypoints[0].x * img.shape[0]
                right_eye_y = detection.location_data.relative_keypoints[1].x * img.shape[0]
                
                # draw the left eye and right eye
                cv2.circle(img, (int(left_eye_x), int(left_eye_y)), 5, (0,0,255), -1)
                cv2.circle(img, (int(right_eye_x), int(right_eye_y)), 5, (0,0,255), -1)
                
                # draw the line b/w eyes
                cv2.line(img, (int(left_eye_x), int(left_eye_y)), (int(right_eye_x), int(right_eye_y)), (0, 0, 255), 2)
                
                # calc distance b/w eyes in pixels
                w = math.dist([left_eye_x, left_eye_y], [right_eye_x, right_eye_y])
                
                # Calc distance from person to camera
                D = (W * F)/ w
                # print("Distance from cam to per:", D)
                cv2.putText(img, f'{(int(D))} cm',
                            (int(left_eye_x + (right_eye_x - left_eye_x)/2), int(left_eye_y + (right_eye_y-left_eye_y)/2)),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score, D])
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
    
    detector = FaceDetector()
    filename = "facesFoundInVideo_Encodings.pickle"
    time.sleep(3.0)
    print("[INFO] Reading faces. Press 'q' to stop.")
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
    fps1 = FPS().start()
    while cap.isOpened():
        success, img = cap.read()
        # print("Image shape: ", img.shape)
        print("[INFO] Detecting faces... ")

        # Resize (zoom into) the image
        # img = detector.zoom_image(img, zoom_factor=1.5)

        img, bboxs = detector.findFaces(img)
        # print(bboxs) # [[0, (608, 258, 245, 245), [0.9235379695892334]]]
        
        if (len(bboxs)!=0):
            try:
                # compute the facial embedding for the face
                print("[INFO] Encoding faces... ")
                (startX, startY, endX, endY) = bboxs[0][1]
                distance = bboxs[0][3]
                print("Length:",len(bboxs), distance)
                
                # extract the face ROI and grab the ROI dimensions
                # Ensure startX is smaller than endX and startY is smaller than endY
                startX, endX = min(startX, endX), max(startX, endX)
                startY, endY = min(startY, endY), max(startY, endY)

                face = img[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                # print("Height: ", fH, "Width: ", fW)
                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue
                # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                encodings = embedder.forward()
                """
                foundEncodings = [] # initialize the list of found encodings
                
                # print("Encodings:",encodings.shape)
                # loop over the encodings (Here we will pass the encodings to the server, but for now we will just store them in a list)
                for encoding in encodings:
                    foundEncodings.append(encoding)
                """
                
                print("FE:", len(encodings))
                """
                # Check if the file exists and load its data if it does
                if os.path.exists(filename):
                    with open(filename, "rb") as f:
                        old_data = pickle.load(f)
                        # print("Old data: ", len(old_data["encodings"]))
                        foundEncodings.extend(old_data["encodings"])

                # Only keep the last 10 encodings
                final_encodings = foundEncodings[-10:]
                # print("Final Enc:", len(final_encodings))

                # Update or create the file with the appended data
                data = {"encodings": final_encodings}
                with open(filename, "wb") as f:
                    f.write(pickle.dumps(data))

                """

                print("[INFO] Done encoding faces. ")

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
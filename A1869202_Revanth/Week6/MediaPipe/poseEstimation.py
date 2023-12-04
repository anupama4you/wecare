# Code written by me
import mediapipe as mp
import cv2
import imutils
import time
from imutils.video import VideoStream
from imutils.video import FPS

mp_pose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Initialize video stream using imutils.
vs = VideoStream(src=0).start()
time.sleep(2.0)
pTime = 0
fps1 = FPS().start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=840, height=640)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = pose.process(rgb_frame)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    
    # update the FPS counter
    fps1.update()

    # Check if landmarks were detected before drawing or processing
    if result.pose_landmarks:
        mpDraw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(result.pose_landmarks.landmark):
            x = int(lm.x * 640)
            y = int(lm.y * 480)
            cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)
            cv2.putText(frame, str(id), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
   
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

fps1.stop()
print("[INFO] elasped time: {:.2f}".format(fps1.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps1.fps()))

# Clean up
cv2.destroyAllWindows()
vs.stop()
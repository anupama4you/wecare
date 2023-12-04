import cv2
import time
import imutils
from imutils.video import VideoStream 
from imutils.video import FPS

counter = 0
imageTimeCounter = 0.0
modelTimeCounter = 0.0
iterTimeCounter = 0.0

print("[INFO] loading model...")
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

font = cv2.FONT_HERSHEY_SIMPLEX

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
while True:
    # Capture frame-by-frame
    # overall_start = time.time() 
    # start_time = time.time() # timer start 1
    
    # read frames
    frame = vs.read()
    frame = imutils.resize(frame, width=300)

    """
    end_time = time.time()
    elapsed_time = end_time - start_time
    imageTimeCounter += elapsed_time
    print(f"[INFO] Time taken to get image: {elapsed_time} seconds")
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #start_time = time.time() # timer start 2
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    """
    end_time = time.time()
    elapsed_time = end_time - start_time
    modelTimeCounter += elapsed_time
    print(f"[INFO] Time taken for model to detect: {elapsed_time} seconds") # timer end 2
    """

    # start_time = time.time()
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame,'Face',(x, y), font, 1, (0, 0, 255), 2)

    """
    end_time = time.time()
    elapsed_time = end_time - start_time
    iterTimeCounter += elapsed_time
    print(f"[INFO] Time taken for model to iterate: {elapsed_time} seconds")
    """

    # cv2.putText(frame,'Number of Faces : ' + str(len(faces)),(40, 40), font, 1,(0,0,255),2)      
    
    # update the FPS counter
    fps.update()
    #counter += 1
    
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# overall_end = time.time()
fps.stop()
#imageTimeCounter = imageTimeCounter/float(counter)
#modelTimeCounter = modelTimeCounter/float(counter)
#iterTimeCounter = iterTimeCounter/float(counter)
#print(f"[INFO] Average time taken to get image: {imageTimeCounter} seconds")
#print(f"[INFO] Average time taken to get model: {modelTimeCounter} seconds")
#print(f"[INFO] Average time taken to finish iterations: {iterTimeCounter} seconds")
#print(f"[INFO] Overall FPS: {(counter)/(overall_end-overall_start)} fps")
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# When everything is done, release the capture
cv2.destroyAllWindows()
vs.stop()
from imutils.video import VideoStream
import numpy as np 
import argparse
import imutils
import cv2
import threading
import queue
import time

exitSignal = False

def captureResizeBlob(vs, blobQueue):
    global exitSignal
    
    while not exitSignal:
        startTime = time.time()
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        blobQueue.put((blob, frame))
        endTime = time.time()
        print(f"Thread 1 Execution Time: {endTime - startTime} seconds")

def applyModel(net, blobQueue, detectionQueue):
    global exitSignal
    
    while not exitSignal:
        startTime = time.time()
        blob, frame = blobQueue.get()
        if blob is None:
            break
        net.setInput(blob)
        detections = net.forward()
        detectionQueue.put((frame, detections))
        endTime = time.time()
        print(f"Thread 2 Execution Time: {endTime - startTime} seconds")

def processAndDisplay(detectionQueue, args, vs):
    global exitSignal
    counter = 0
    start = time.time()
    
    while not exitSignal:
        startTime = time.time()
        frame, detections = detectionQueue.get()
        if frame is None:
            break
        (h, w) = frame.shape[:2]
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < args["confidence"]:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            exitSignal = True
            break

        counter += 1

        endTime = time.time()
        print(f"Thread 3 Execution Time: {endTime - startTime} seconds")
    
    end = time.time()
    totalTime = end - start
    if totalTime > 0:
        print(f"FPS: {counter / totalTime}")

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()

    blobQueue = queue.Queue(maxsize=10)
    detectionQueue = queue.Queue(maxsize=10)

    thread1 = threading.Thread(target=captureResizeBlob, args=(vs, blobQueue))
    thread2 = threading.Thread(target=applyModel, args=(net, blobQueue, detectionQueue))
    thread3 = threading.Thread(target=processAndDisplay, args=(detectionQueue, args, vs))

    thread1.start()
    thread2.start()
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()

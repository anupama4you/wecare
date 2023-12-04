import numpy as np
import argparse
import imutils
from imutils.video import FPS
from imutils.video import VideoStream
import time
import cv2
from threading import Thread
from queue import Queue

# Global variables
exitSignal = False
frameTimes = []

def captureResizeBlob(vs, blobQueue):
    global exitSignal
    while not exitSignal:
        # grab the frame from the threaded video stream and resize it
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        blobQueue.put((frame, blob))

    # signify that we have finished processing
    blobQueue.put((None, None))

def applyModel(net, blobQueue, detectionQueue):
    global exitSignal
    while not exitSignal or not blobQueue.empty():
        frame, blob = blobQueue.get()
        if blob is None:
            break

        net.setInput(blob)
        detections = net.forward()

        detectionQueue.put((frame, detections))

    # signify that we have finished processing
    detectionQueue.put((None, None))

def processAndDisplay(detectionQueue, args, vs):
    global exitSignal
    global frameTimes
    while not exitSignal:
        startTime = time.time()

        frame, detections = detectionQueue.get()
        if frame is None:
            break

        if detections is not None:
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

        endTime = time.time()
        frameTimes.append(endTime - startTime)

    # After loop, print per-frame times
    for idx, t in enumerate(frameTimes, 1):
        print(f"Frame {idx} Time: {t} seconds")

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
    # argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load the model
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # initialize video stream
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # create queues
    blobQueue = Queue(maxsize=10)
    detectionQueue = Queue(maxsize=10)

    # create threads
    thread1 = Thread(target=captureResizeBlob, args=(vs, blobQueue,))
    thread2 = Thread(target=applyModel, args=(net, blobQueue, detectionQueue,))
    thread3 = Thread(target=processAndDisplay, args=(detectionQueue, args, vs,))

    thread1.start()
    thread2.start()
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()

    print("All tasks completed!")

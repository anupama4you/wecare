import cv2
import argparse
import numpy as np
import imutils
from imutils.video import FPS, VideoStream
import time
from multiprocessing import Process, Queue, Value, current_process

def capture_and_resize(blob_queue, exit_signal):
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    
    while not exit_signal.value:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        blob_queue.put((blob, frame))

    vs.stop()

def apply_model(blob_queue, detection_queue, exit_signal):
    while not exit_signal.value:
        if not blob_queue.empty():
            blob, frame = blob_queue.get()
            net.setInput(blob)
            detections = net.forward()
            detection_queue.put((detections, frame))

def detect_and_show(detection_queue, exit_signal, args):
    while not exit_signal.value:
        if not detection_queue.empty():
            detections, frame = detection_queue.get()

            (h, w) = frame.shape[:2]
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < args["confidence"]:
                    continue

                # Compute bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box on the frame
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            # Show the frame
            cv2.imshow("Frame", frame)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    blob_queue = Queue()
    detection_queue = Queue()
    exit_signal = Value('b', False)

    process1 = Process(target=capture_and_resize, args=(blob_queue, exit_signal))
    process2 = Process(target=apply_model, args=(blob_queue, detection_queue, exit_signal))
    process3 = Process(target=detect_and_show, args=(detection_queue, exit_signal, args))

    process1.start()
    process2.start()
    process3.start()

    fps = FPS().start()

    try:
        while True:
            time.sleep(0.1)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            fps.update()
    except KeyboardInterrupt:
        pass

    exit_signal.value = True

    process1.join()
    process2.join()
    process3.join()

    fps.stop()
    print(f"[INFO] elapsed time: {fps.elapsed()}")
    print(f"[INFO] approx. FPS: {fps.fps()}")

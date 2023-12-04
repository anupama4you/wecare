# CMD: python3 FaceDetection_Video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from imutils.video import VideoStream # VideoStream from imutils.video for capturing video streams.
import numpy as np 
import argparse
import imutils # imutils for various image processing functions - resizing, rotating, etc.
from imutils.video import FPS
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
counter = 0
imageTimeCounter = 0.0
modelTimeCounter = 0.0
iterTimeCounter = 0.0

fps = FPS().start()
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	# start_time = time.time()
	frame = vs.read()
	frame = imutils.resize(frame, width=300)
	"""
	end_time = time.time()
	elapsed_time = end_time - start_time
	imageTimeCounter += elapsed_time
	print(f"[INFO] Time taken to get image: {elapsed_time} seconds")
	"""

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]

	# start_time = time.time()
	blob = cv2.dnn.blobFromImage(frame, 1.1,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	
	"""
	end_time = time.time()
	elapsed_time = end_time - start_time
	modelTimeCounter += elapsed_time
	print(f"[INFO] Time taken for model to detect: {elapsed_time} seconds")
	"""

	# start_time = time.time()
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]
	
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < args["confidence"]:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	"""
	end_time = time.time()
	elapsed_time = end_time - start_time
	iterTimeCounter += elapsed_time
	print(f"[INFO] Time taken for model to iterate: {elapsed_time} seconds")
	"""

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	
	# adjusting to 20FPS. = 1/20 = 0.05; 12.95 FPS: 1/12.95 = 0.077; 10 FPS = 1/10 = 0.1
	# time.sleep(1)

	# update the FPS counter
	fps.update()
	# counter += 1

# stop the timer and display FPS information
fps.stop()
"""
imageTimeCounter = imageTimeCounter/float(counter)
modelTimeCounter = modelTimeCounter/float(counter)
iterTimeCounter = iterTimeCounter/float(counter)
print(f"[INFO] Average time taken to get image: {imageTimeCounter} seconds")
print(f"[INFO] Average time taken to get model: {modelTimeCounter} seconds")
print(f"[INFO] Average time taken to finish iterations: {iterTimeCounter} seconds")
"""
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
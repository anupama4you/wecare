import cv2
import numpy as np
import argparse
import os

# Command-line argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

font = cv2.FONT_HERSHEY_SIMPLEX

cascPath1 = "haarcascade_frontalface_default.xml"
cascPath2 = "haarcascade_frontalface_alt_tree.xml"
cascPath3 = "haarcascade_frontalface_alt.xml"
cascPath4 = "haarcascade_frontalface_alt2.xml"

faceCascade_1 = cv2.CascadeClassifier(cascPath1)
faceCascade_2 = cv2.CascadeClassifier(cascPath2)
faceCascade_3 = cv2.CascadeClassifier(cascPath3)
faceCascade_4 = cv2.CascadeClassifier(cascPath4)

cascades = [faceCascade_1, faceCascade_2, faceCascade_3, faceCascade_4]

# Define colors for each model
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

def detect_and_draw_faces(image_path):
    image = cv2.imread(image_path, 1)  # Load in color
    height, width, _ = image.shape

    for idx, faceCascade in enumerate(cascades, 1):
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), colors[idx-1], 3)

    # Add legend at top right corner
    legend_x = width - 200
    for i, color in enumerate(colors, 1):
        cv2.putText(image, f"Model {i}", (legend_x, 35 * i), font, 0.7, color, 2)

    return image

# Path to the test image
image_path = "../test/" + args["image"]
result_image = detect_and_draw_faces(image_path)

# Check if the "outputs" directory exists; if not, create it
output_directory = "outputs"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Extract the filename from the image path and save to "outputs" directory
filename = os.path.basename(args["image"])
output_path = os.path.join(output_directory, filename)
cv2.imwrite(output_path, result_image)

# Display the result image
cv2.imshow('Faces Detected', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

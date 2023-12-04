import argparse
import sys
import time
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

# Constants
COLORS = [(0, 0, 255)]  # Red for visualization
FONT_SIZE = 1
FONT_THICKNESS = 1
LEFT_MARGIN = 24
ROW_SIZE = 20

def preprocess_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def run(model_path, camera_id, frame_size, num_threads):
    # Initialize the webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        sys.exit('ERROR: Unable to initialize webcam.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

    # Load the TFLite model
    options = vision.ObjectDetectorOptions(
        base_options=core.BaseOptions(file_name=model_path, num_threads=num_threads),
        detection_options=processor.DetectionOptions(max_results=3, score_threshold=0.3)
    )
    detector = vision.ObjectDetector.create_from_options(options)

    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame = cv2.flip(frame, 1)
        rgb_image = preprocess_image(frame)
        tensor_image = vision.TensorImage.create_from_array(rgb_image)
        detection_result = detector.detect(tensor_image)
        frame = utils.visualize(frame, detection_result)

        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (LEFT_MARGIN, ROW_SIZE), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, COLORS[0], FONT_THICKNESS)
        cv2.imshow('Object Detection', frame)

        # Exit on pressing 'ESC'
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Real-time object detection using TFLite", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Path of the object detection model.', default='efficientdet_lite0.tflite')
    parser.add_argument('--cameraId', help='ID of the camera to use.', type=int, default=0)
    parser.add_argument('--frameSize', help='Size of the camera frame as (width, height).', type=lambda s: tuple(map(int, s.split(','))), default=(640, 480))
    parser.add_argument('--numThreads', help='Number of CPU threads to run the model on.', type=int, default=4)
    args = parser.parse_args()

    run(args.model, args.cameraId, args.frameSize, args.numThreads)

if __name__ == "__main__":
    main()

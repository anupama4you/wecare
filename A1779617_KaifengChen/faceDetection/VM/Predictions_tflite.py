import time
import tensorflow as tf
import numpy as np
import os
import argparse
from pathlib import Path
from timeit import default_timer as timer
import numpy as np
from PIL import Image


def preprocess_image(image_path, target_size):
    # Use PIL to load the image
    image = Image.open(image_path)

    # Resize the image
    image = image.resize(target_size[:-1])

    # Convert to numpy array
    image_array = np.array(image).astype(np.float32)

    # Normalize
    image_array = image_array / 255.0

    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def load_tflite_model(model_name):
    # load the tflite model
    tflite_model = Path(model_name).read_bytes()
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    return interpreter


def Predictions(patients_room_no: int, test_image: str, model_name: str = "MOBILENETV2", target_size: tuple = (96, 96, 3), Patient_images_folder: str = "Patients_Images"):
    """
    This function takes the path to the patients room folder and the path to the model and performs predictions

    params:
        patients_room_dir: Path
            The directory where the cropped face images npy file is stored of that patient 
            (1 - room per patient, so the folder name is the patient name/room number)
        test_image: image
            test image to be compared
        model_path: str
            The path to the trained model (saved as .h5 file)
    """

    interpreter = load_tflite_model(model_name+"_SIAMESE.tflite")

    # -----> We can convert in future from here to run with the firebase data <-----

    # Load the numpy array of the patient room
    while True:
        try:
            patients_images_in_given_room = np.load(os.path.join(
                Patient_images_folder, str(patients_room_no)+".npy"))
            break
        except:
            time.sleep(0.3)

    start_time = timer()
    # Resize the test image to the target size
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.array([test_image])  # Adds batch dimension
    test_image = test_image / 255.0

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    no_of_times_matched = 0
    for idx, image in enumerate(iter(patients_images_in_given_room)):
        image = np.array([image])

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.set_tensor(input_details[1]['index'], test_image)

        # Run the inference
        interpreter.invoke()

        # Retrieve the output tensor
        prediction = interpreter.get_tensor(
            output_details[0]['index']).squeeze()

        if prediction > 0.5:
            no_of_times_matched += 1

    end_time = timer()
    print(
        f"[INFO] Given Person has Matched with patient {patients_room_no} by: {no_of_times_matched}/{len(patients_images_in_given_room)} times = {no_of_times_matched/len(patients_images_in_given_room)*100}% | Time taken: {end_time-start_time} seconds")

    # Return the result of the prediction
    return ((no_of_times_matched/len(patients_images_in_given_room))*100.0)


if __name__ == "__main__":
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--Patient_images_folder", required=False,
                    help="Path to the patients images folder", default="Patients_Images")
    ap.add_argument("-r", "--patients_room_no", required=True,
                    help="Patient's room number")
    ap.add_argument("-m", "--model_name", required=False,
                    help="Path to the trained model", default="MOBILENETV2")
    ap.add_argument("-s", "--target_size", required=False,
                    help="desired height/width shape of the image, For Model: Custom: 64 -> (64, 64, 3), MobileNetV2: 224 -> (224, 224, 3), for others: 300 -> (300, 300, 3)", default=96)
    ap.add_argument("-t", "--test_image", required=True,
                    help="Path to the test image")
    args = vars(ap.parse_args())

    # Call the Predictions function
    Predictions(args["patients_room_no"], args["test_image"], args["model_name"], target_size=(int(
        args["target_size"]), int(args["target_size"]), 3), Patient_images_folder=args["Patient_images_folder"])

# Command to run this file:
# python Predictions_tflite.py -r 1 -t TestImages/7/1.jpg

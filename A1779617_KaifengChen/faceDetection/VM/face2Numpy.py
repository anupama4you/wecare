import tensorflow as tf
import numpy as np
import os
import argparse
from pathlib import Path
import numpy as np


def Faces2Numpy(patients_dir: Path, target_size: tuple = (96, 96, 3)) -> np.ndarray:
    """
    Converts the cropped face images into a numpy array files (resized to 224x224x3, 1/255.0 normalized)

    params:
        patients_room_dir: Path
            The directory where the cropped face images are stored of that patient 
            (1 - room per patient, so the folder name is the patient name/room number)

    returns:
        Returns the numpy array
    """

    # Iterate over patients folder patients_images/patient_room_no/0..n images
    for patient in patients_dir.iterdir():
        # if file exits, skip or if anything other than folder, skip
        if not os.path.exists(os.path.join(patients_dir, patient.name+".npy")) and patient.is_dir():
            # Iterate over the images in the patients room folder
            images = []
            for image in patient.iterdir():
                # Read, resize, convert to numpy array and normalize the image
                img = tf.keras.preprocessing.image.load_img(
                    image, target_size=target_size)
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = img / 255.0

                # Append the image to the images list, and the label to the labels list
                images.append(img)

            # Convert the images list to numpy array
            images = np.array(images)

            # Save the numpy array in patients_dir/{patient}.npy save as room_no.npy
            np.save(patients_dir / (patient.name + ".npy"), images)


if __name__ == "__main__":
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--Patient_images_folder", required=False,
                    help="Path to the patients images folder", default="Patients_Images")
    ap.add_argument("-s", "--target_size", required=False,
                    help="desired height/width shape of the image, For Model: Custom: 64 -> (64, 64, 3), MobileNetV2: 224 -> (224, 224, 3), for others: 300 -> (300, 300, 3)", default=96)
    args = vars(ap.parse_args())

    # Convert the cropped face images to numpy arrays
    Faces2Numpy(Path(args["Patient_images_folder"]), target_size=(
        int(args["target_size"]), int(args["target_size"]), 3))

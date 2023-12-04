import tensorflow as tf
import numpy as np
import os
from pathlib import Path

# In this file in the given directory, we will convert the given "cropped face" images into a numpy array and save it as a .npy file (resized to 64x64x3, 1/255.0 normalized)

def Faces2Numpy(source_dir: Path, saving_dir: Path, target_size:tuple = (64, 64, 3)):
    """
    Converts the cropped face images into a numpy array files (resized to 64x64x3, 1/255.0 normalized)

    params:
        source_dir: Path
            The directory where the cropped face images are stored
        saving_dir: Path
            The directory where the .npy file will be saved
    """
    

    # Iterate over the data folder (train and test)
    for folder in source_dir.iterdir():
        # Create a list to store the images and labels
        images = []
        labels = []
        # Iterate over the person folders in the train/test folder
        for person_folder in folder.iterdir():
            # Iterate over the images in the person folder
            for image in person_folder.iterdir():
                # Read, resize, convert to numpy array and normalize the image
                img = tf.keras.preprocessing.image.load_img(image, target_size=target_size)
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = img / 255.0
                
                # Append the image to the images list, and the label to the labels list
                images.append(img)
                labels.append(person_folder.name)
        
        # Save the images and labels list as .npy file
        np.save(saving_dir / f"{folder.name}_images.npy", images)
        np.save(saving_dir / f"{folder.name}_labels.npy", labels)
    
        # Load the train and test images and labels
        images = np.load(saving_dir / f"{folder.name}_images.npy")
        labels = np.load(saving_dir / f"{folder.name}_labels.npy")
        print(f"[INFO] {folder.name} images shape: {images.shape}, labels shape: {labels.shape}")
    
    print("[INFO] Sucessfully converted cropped face images to numpy arrays!")

if __name__ == "__main__":
    # Define the source and saving directory
    source_dir = Path("data/")
    saving_dir = Path("data_numpy/")
    # Create the saving directory if it doesn't exist
    if not saving_dir.exists():
        saving_dir.mkdir()
    # Call the Faces2Numpy function
    Faces2Numpy(source_dir, saving_dir)
            
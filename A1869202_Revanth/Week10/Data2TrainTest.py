# This converts the given folder images data to train and test data.
import os
import shutil
from random import shuffle
from pathlib import Path

def Data2TrainTest(source_dir:Path, train_target_dir:Path, test_target_dir:Path, train_ratio:float):
    # Create train and test directories if they don't exist
    # save them under data/train_target_dir, data/test_target_dir folders
    FolderName = "data/"
    train_target_dir = os.path.join(FolderName, train_target_dir)
    test_target_dir = os.path.join(FolderName, test_target_dir)
    os.makedirs(train_target_dir, exist_ok=True)
    os.makedirs(test_target_dir, exist_ok=True)

    # List all the person folders
    persons = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    # only keep randomly selected 10 classes
    shuffle(persons)

    print("[INFO] Splitting data into Train & Test...")
    for person in (persons):
        # List all images of the person
        images = [f for f in os.listdir(os.path.join(source_dir, person)) if f.endswith(('.png', '.jpg', '.jpeg'))]
        shuffle(images)

        # Calculate split index
        split_idx = int(train_ratio * len(images))
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        # Define person-specific directories in train and test
        train_person_dir = os.path.join(train_target_dir, person)
        test_person_dir = os.path.join(test_target_dir, person)

        os.makedirs(train_person_dir, exist_ok=True)
        os.makedirs(test_person_dir, exist_ok=True)

        # Rename images to respective directories, a copy of faces in the train and test folders
        for idx, img in enumerate(train_images):
            new_img_name = str(idx) + os.path.splitext(img)[-1]  # new name with extension
            # Create a copy of the image in the train folder
            shutil.copy(os.path.join(source_dir, person, img), os.path.join(train_person_dir, new_img_name))

        for idx, img in enumerate(test_images):
            new_img_name = str(idx) + os.path.splitext(img)[-1]  # new name with extension
            shutil.copy(os.path.join(source_dir, person, img), os.path.join(test_person_dir, new_img_name))
    
    print(f"[INFO] Splitting Done: {len(persons)} persons data splitted into {train_ratio*100:.1f}% train and {(1-train_ratio)*100:.1f}% test data")

    return FolderName
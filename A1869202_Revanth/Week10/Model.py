import shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
import argparse
import Faces2Numpy
from timeit import default_timer as timer

def data_generator(data_dir:Path, mode:str, batch_size:int = 40):
    """
    This function creates data generators for the given data directory.

    Params:
        data_dir: Path to the data directory
        mode: Mode of the program, train or test
        batch_size: Batch size for the data generators
    
    Returns:
        image_data_gen: Image data generator, A dictionary of train and test data generators.
        It Can be accessed as image_data_gen['train'] and image_data_gen['test']
    """
    INPUT_SHAPE = (224, 224) # (300, 300) for MobileNetV2, others (224, 224)

    if mode == "train":
        # Define the data transformations using ImageDataGenerator
        data_generators = {
            'train': ImageDataGenerator(
                rescale=1./255,  # Rescaling
                rotation_range=20,  # Random rotations for data augmentation
                width_shift_range=0.2,  # Random width shift for data augmentation
                height_shift_range=0.2,  # Random height shift for data augmentation
                horizontal_flip=True  # Random horizontal flips for data augmentation
            ),
            'test': ImageDataGenerator(
                rescale=1./255  # Only rescaling for testing data
            )
        }

        # Create data generators from directory
        image_data_gen = {
            'train': data_generators['train'].flow_from_directory(
                os.path.join(data_dir, 'train'),
                target_size=INPUT_SHAPE, #(300, 300), # (224, 224) for MobileNetV2
                batch_size=batch_size,
                shuffle=True,
                class_mode='categorical'  # Changed to 'categorical' for multi-class
            ),
            'test': data_generators['test'].flow_from_directory(
                os.path.join(data_dir, 'test'),
                target_size=INPUT_SHAPE,
                batch_size=batch_size,
                shuffle=True,
                class_mode='categorical'  # Changed to 'categorical' for multi-class
            )
        }

        class_names = list(image_data_gen['train'].class_indices.keys())
        print(f"[INFO] There are {len(class_names)} classes.")

        return image_data_gen
    
    else:
        # Define the data transformations using ImageDataGenerator
        data_generators = {
            'test': ImageDataGenerator(
                rescale=1./255  # Only rescaling for testing data
            )
        }

        # Create data generators from directory
        image_data_gen = {
            'test': data_generators['test'].flow_from_directory(
                os.path.join(data_dir, 'test'),
                target_size=INPUT_SHAPE,
                batch_size=batch_size,
                shuffle=True,
                class_mode='categorical'  # Changed to 'categorical' for multi-class
            )
        }

        class_names = list(image_data_gen['test'].class_indices.keys())
        print(f"[INFO] There are {len(class_names)} classes.")

        return image_data_gen

def make_pairs(images_path:Path, labels_path:Path):
    """
    This function creates pairs of images and labels. It is used for Siamese network.

    Params:
        images_path: Path to the images.npy file
        labels_path: Path to the labels.npy file
    
    Returns:
        pair_images: Pair of images
        pair_labels: Pair of labels
    """

    # Load the images and labels
    images_dataset = np.load(images_path)
    labels_dataset = np.load(labels_path)

    unique_labels = np.unique(labels_dataset)
    label_wise_indices = dict()
    for label in unique_labels:
        label_wise_indices.setdefault(label,
                                      [index for index, curr_label in enumerate(labels_dataset) if
                                       label == curr_label])
    
    pair_images = []
    pair_labels = []
    for index, image in enumerate(images_dataset):
        pos_indices = label_wise_indices.get(labels_dataset[index])
        pos_image = images_dataset[np.random.choice(pos_indices)]
        pair_images.append((image, pos_image))
        pair_labels.append(1)

        neg_indices = np.where(labels_dataset != labels_dataset[index])
        neg_image = images_dataset[np.random.choice(neg_indices[0])]
        pair_images.append((image, neg_image))
        pair_labels.append(0)

    return np.array(pair_images), np.array(pair_labels)
    
def model(model_name:str, out_features:int = 256, CONVERT_MODEL_2_SIAMESE:bool = True, input_shape:tuple = (224, 224, 3)): # (224, 224) for MobileNetV2, others (300, 300)
    """
    This function creates a model based on the given model name.

    Params:
        model_name: Name of the model to be created
        out_features/embedding_dim: Number of output features/embedding dimension
        input_shape: Input shape of the model
        embedding_dim: Dimension of the embedding layer
        CONVERT_MODEL_2_SIAMESE: Convert the model to a siamese network or not
    
    Returns:
        model: The created model

    Available models:
        > VGG19, VGG16_A, VGG16_B, INCEPTIONV3, MOBILENETV2, RESNET101V2, CUSTOM 
    """

    model = keras.models.Sequential()

    if model_name == "VGG19":
        # VGG19

        model.add(tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape))

        # Other layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(out_features, activation='relu'))
        
        if CONVERT_MODEL_2_SIAMESE != True:
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(out_features, activation='softmax'))

        # Set TL layers not trainable
        model.layers[0].trainable = False
    
    elif model_name == "VGG16_A":
        # VGG16
        model.add(tf.keras.applications.VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape))

        # Other layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(out_features, activation='relu'))
        
        if CONVERT_MODEL_2_SIAMESE != True:
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(out_features, activation='softmax'))

        # Set TL layers not trainable
        model.layers[0].trainable = False
    
    elif model_name == "VGG16_B":
        # VGG16
        model.add(tf.keras.applications.VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape))

        # Other layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(out_features, activation='relu'))

        if CONVERT_MODEL_2_SIAMESE != True:
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(out_features, activation='softmax'))

        # Set TL layers not trainable
        model.layers[0].trainable = False

    elif model_name == "INCEPTIONV3":
        # INCEPTION V3

        model.add(tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape))

        # Other layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(out_features, activation='relu'))
        
        if CONVERT_MODEL_2_SIAMESE != True:
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(out_features, activation='softmax'))

        # Set TL layers not trainable
        model.layers[0].trainable = False
    
    elif model_name == "MOBILENETV2":
        
        model.add(tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape)) # Only for this network

        # Other layers
        # Other layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(out_features, activation='relu'))
        
        if CONVERT_MODEL_2_SIAMESE != True:
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(out_features, activation='softmax'))

        # Set TL layers not trainable
        model.layers[0].trainable = False
    
    elif model_name == "RESNET50V2":

        model.add(tf.keras.applications.ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape))

        # Other layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(out_features, activation='relu'))

        if CONVERT_MODEL_2_SIAMESE != True:
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(out_features, activation='softmax'))

        # Set EfficientNet layers not trainable
        model.layers[0].trainable = False

    elif model_name == "RESNET101V2":
        
        model.add(tf.keras.applications.ResNet101V2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape))

        # Other layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(out_features, activation='relu'))

        if CONVERT_MODEL_2_SIAMESE != True:
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(out_features, activation='softmax'))

        # Set EfficientNet layers not trainable
        model.layers[0].trainable = False
    
    elif model_name == "DENSENET121":
            
            model.add(tf.keras.applications.DenseNet121(
                include_top=False,
                weights="imagenet",
                input_shape=input_shape))
    
            # Other layers
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(1024, activation='relu'))
            model.add(keras.layers.Dense(1024, activation='relu'))
            model.add(keras.layers.Dense(512, activation='relu'))
            model.add(keras.layers.Dense(out_features, activation='relu'))
    
            if CONVERT_MODEL_2_SIAMESE != True:
                model.add(keras.layers.Dropout(0.2))
                model.add(keras.layers.Dense(out_features, activation='softmax'))
    
            # Set EfficientNet layers not trainable
            model.layers[0].trainable = False

    elif model_name == "CUSTOM":
        # CUSTOM MODEL

        model = keras.models.Sequential()

        # Input layer
        model.add(keras.layers.InputLayer(input_shape=input_shape))

        # First convolution block
        model.add(keras.layers.Conv2D(64, (10, 10), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.3))

        # Second convolution block
        model.add(keras.layers.Conv2D(128, (7, 7), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.3))

        # Third convolution block
        model.add(keras.layers.Conv2D(128, (4, 4), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.3))

        # Fourth convolution block
        model.add(keras.layers.Conv2D(256, (4, 4), padding="same", activation="relu"))

        # Fully connected layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(4096, activation="relu"))
        model.add(keras.layers.Dense(1024, activation="relu"))
        model.add(keras.layers.Dense(out_features, activation="sigmoid")) # Returns the 1024D embeddings

    else:
        print("[ERROR] Invalid model name!")

    # SIAMESE NETWORK 

    if CONVERT_MODEL_2_SIAMESE == True:
        imgA = keras.layers.Input(shape=input_shape)
        imgB = keras.layers.Input(shape=input_shape)
        featureExtractor = model
        featsA = featureExtractor(imgA)
        featsB = featureExtractor(imgB)

        # Add a customized distance layer to compute the distance between the two encodings (Euclidean distance)
        distance = keras.layers.Lambda(lambda tensors: tf.keras.backend.abs(tensors[0] - tensors[1]))

        # Add a dense layer with a sigmoid unit to generate the similarity score
        similarity_layer = keras.layers.Dense(1, activation='sigmoid')

        # Combine the network
        distance_layer = distance([featsA, featsB])
        output_layer = similarity_layer(distance_layer)

        # Convert to a model
        siamese_model = keras.models.Model(inputs=[imgA, imgB], outputs=output_layer)
    
    else:
        siamese_model = model
    
    return siamese_model

def train(model, paired_train_images:np.ndarray, paired_train_labels:np.ndarray, model_name:str, epochs:int = 10, patience:int = 7, learning_rate:float = 0.001, batch_size:int = 40, validation_split:float = 0.1):
    """
    This function trains the given model.

    Params:
        model: Model to be trained
        paired_train_images: Paired train images
        paired_train_labels: their labels (0 - different, 1 - same)
        model_name: Name of the model
        epochs: Number of epochs
        patience: Patience for early stopping
        learning_rate: Learning rate
        batch_size: Batch size
        validation_split: Validation split
        reg_strength: Regularization strength
    
    Returns:
        history: Training history
    """

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    # EarlyStopping callback
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)

    # ModelCheckpoint callback
    model_name = model_name + "_SIAMESE.h5"
    model_checkpoint_cb = keras.callbacks.ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True)

    # Combining all callbacks into a list
    callbacks = [early_stopping_cb, model_checkpoint_cb]

    # Training
    history = model.fit([paired_train_images[:, 0], paired_train_images[:, 1]], paired_train_labels[:], validation_split=validation_split, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

    print("[INFO] Training done!")
    return history

def test(model, paired_test_images:np.ndarray, paired_test_labels:np.ndarray):
    """
    This function tests the given model.

    Params:
        model: Model to be tested
        image_data_gen: Image data generator
    
    Returns:
        history: Testing history
    """

    # Testing
    test_loss, test_acc = model.evaluate([paired_test_images[:, 0], paired_test_images[:, 1]], paired_test_labels[:])
    print("[INFO] Model: Test Loss:", test_loss, "Test Accuracy:", test_acc)
    print("[INFO] Testing done!")
    return test_loss, test_acc

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data_dir", required=False,  default="data",
                help="path to the data directory for train, test mode. Single image path for predict mode")
    ap.add_argument("-n", "--model_name", required=True,
        help="name of the model to be created, available models: VGG19, VGG16_A, VGG16_B, INCEPTIONV3, MOBILENETV2, RESNET50V2, RESNET101V2, DENSENET121, CUSTOM")
    ap.add_argument("-m", "--mode", required=True,
        help="mode of the program, train or test or predict or emb - get the embedding model from the siamese model")
    ap.add_argument("-e", "--epochs", required=False,
        help="number of epochs", default=10)
    ap.add_argument("-p", "--patience", required=False,
        help="patience for early stopping", default=7)
    ap.add_argument("-s", "--input_shape", required=False,
                    help="input shape of the model", default=224)
    ap.add_argument("-lr", "--learning_rate", required=False,
        help="learning rate", default=0.001)
    args = vars(ap.parse_args())

    # Check if the given data_dir exists
    if not os.path.exists(args["data_dir"]):
        print(f"[ERROR] The given data_dir {args['data_dir']} doesn't exist!")
        exit()
    
    # Check if the given model_name is valid
    if args["model_name"] not in ["VGG19", "VGG16_A", "VGG16_B", "INCEPTIONV3", "MOBILENETV2", "RESNET50V2", "RESNET101V2", "DENSENET121", "CUSTOM"]:
        print(f"[ERROR] The given model_name {args['model_name']} is invalid!")
        exit()
    
    # Check if the given mode is valid
    if args["mode"] not in ["train", "test", "predict", "emb"]:
        print(f"[ERROR] The given mode {args['mode']} is invalid!")
        exit()
    
    # Create/load model
    if args["mode"] == "train":
        # Create data pairs
        print("[INFO] Creating data pairs...")
        paired_train_images, paired_train_labels = make_pairs(Path(args["data_dir"] + "/train_images.npy"), Path(args["data_dir"] + "/train_labels.npy"))
        paired_test_images, paired_test_labels = make_pairs(Path(args["data_dir"] + "/test_images.npy"), Path(args["data_dir"] + "/test_labels.npy"))

        # Create the model with out_features/embedding_dim:int = 1024
        model = model(args["model_name"], out_features=1024, CONVERT_MODEL_2_SIAMESE=True, input_shape=(int(args["input_shape"]), int(args["input_shape"]), 3)) # (224, 224) for MobileNetV2, others (300, 300
        
        # Train model
        history = train(model=model, paired_train_images=paired_train_images, paired_train_labels=paired_train_labels, model_name=args["model_name"], epochs=int(args["epochs"]), patience=int(args["patience"]), learning_rate=float(args["learning_rate"]))

        # Test model (This tests with lastly balanced weights of the model - the model after last epoch, may/may not be the best model)
        test_loss, test_acc = test(model=model, paired_test_images=paired_test_images, paired_test_labels=paired_test_labels)
    
    elif args["mode"] == "test":
        # Load the test data 
        paired_test_images, paired_test_labels = make_pairs(Path(args["data_dir"] + "/test_images.npy"), Path(args["data_dir"] + "/test_labels.npy"))

        # Load the model (This loads the best model, tests on best model)
        model = keras.models.load_model(args["model_name"]+"_SIAMESE.h5")
        
        # Test model
        test_loss, test_acc = test(model=model, paired_test_images=paired_test_images, paired_test_labels=paired_test_labels)
    
    elif args["mode"] == "predict":
        # This takes the image path (data_dir) and predicts the class of the image
        # Extract given patient id from the image path: Patients_Images/data/6/16.jpg
        given_patient_id = args["data_dir"].split("/")[-2]

        # Load the model
        model = keras.models.load_model(args["model_name"]+"_SIAMESE.h5")

        # Load the image
        # print the shape of the image
        image = tf.keras.preprocessing.image.load_img(args["data_dir"], target_size=(int(args["input_shape"]), int(args["input_shape"]), 3)) # (300, 300) for MobileNetV2, others (224, 224)
        input_arr = keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        input_arr = input_arr / 255.0 # Rescale the image
        print("[INFO] Input Image shape after resizing:", input_arr.shape)

        # Predict the class, model takes param 1: set of images, we want to verify matching with & test image
        patients_images_path = Path("Patients_Images")
        if not os.path.exists(patients_images_path):
            print(f"[ERROR] The given patients_images_path {patients_images_path} doesn't exist!")
            exit()

        # if data folder exists inside patients_images_path, dont run some code
        if not os.path.exists(os.path.join(patients_images_path, "data")):
            # Move all the sub-folders in the patients_images_path to the data folder (patients_images_path/0...N) -> patients_images_path/data/0...N)
            for patient in os.listdir(patients_images_path):
                if os.path.isdir(os.path.join(patients_images_path, patient)):
                    shutil.move(os.path.join(patients_images_path, patient), os.path.join("Patients_Images/data/", patient))
            
            Faces2Numpy.Faces2Numpy(Path(patients_images_path), Path(patients_images_path), (int(args["input_shape"]), int(args["input_shape"]), 3))

        images_numpy = np.load("Patients_Images/data_images.npy")
        images_labels = np.load("Patients_Images/data_labels.npy")
        matches = dict()
        start_timer = timer()
        total_counts = dict()
        for idx, image in enumerate(iter(images_numpy)):
            # Predict the class, model takes param 1: image 1, we want to verify matching with & test image
            prediction = model.predict([input_arr, image.reshape(1, int(args["input_shape"]), int(args["input_shape"]), 3)]).squeeze() * 100.0

            # Print the prediction, image number (8.jpg like that)
            if(prediction>=50.0):
                # add the match to the dictionary
                if (images_labels[idx] in matches):
                    # add image only if the probability is higher than the previous one
                    if (matches[images_labels[idx]][1] < prediction):
                        matches[images_labels[idx]] = (matches[images_labels[idx]][0] + 1, prediction, image)
                    else:
                        matches[images_labels[idx]] = (matches[images_labels[idx]][0] + 1, matches[images_labels[idx]][1], matches[images_labels[idx]][2])
                else:
                    matches[images_labels[idx]] = (1, prediction, image)
            
            if(images_labels[idx] in total_counts):
                total_counts[images_labels[idx]] = total_counts[images_labels[idx]] + 1
            else:
                total_counts[images_labels[idx]] = 1
        
        end_timer = timer()
        print(f"[INFO] Time taken to make predictons on all patients: {end_timer - start_timer} seconds")
        # Print the results
        for image_id, (count, prob, _) in matches.items():
            print(f"Id: {given_patient_id} matched with Patient ID: {image_id}, Number of times Matched: {count}/{total_counts[image_id]}, Highest Probability: {prob:.2f}%")
        
        # No matches found
        if (len(matches) == 0):
            print(f"[INFO] No matches found for Patient ID: {given_patient_id}")

    elif args["mode"] == "emb":
        # This takes siaemes model and saves the embedding model
        model = keras.models.load_model(args["model_name"]+"_SIAMESE.h5")

        # Extract the feature extractor from the Siamese model
        feature_extractor = model.get_layer('sequential')

        # Create a new model with a single input image layer that returns the embeddings
        input_image = tf.keras.layers.Input(shape=(224, 224, 3))
        embeddings = feature_extractor(input_image)
        embedding_model = tf.keras.models.Model(inputs=input_image, outputs=embeddings)

        # Save the model
        embedding_model.save(args["model_name"]+"_EMBEDDING.h5")

# Call this file like this:

# With MobileNetv2
# python Model.py -n MOBILENETV2 -m train -d data -s 96 -p 10 -e 50
# python Model.py -n MOBILENETV2 -m test -d data
# python Model.py -n MOBILENETV2 -m predict -d data/test/0/1.jpg -s 96
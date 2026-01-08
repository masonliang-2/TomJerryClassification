import tensorflow as tf
from tensorflow import keras
layers = tf.keras.layers
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple

DATASET_PATH = "tom_and_jerry_training_dataset"
DATASET_TESTING_PATH = "tom_and_jerry_testing_dataset"
IMAGE_SIZE = (64, 64)  # Define the target image size
SINGLE_IMAGE_TESTS_PATH = "single_image_tests"

def clean_dataset(dataset_path) -> None:
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file == ".DS_Store":
                os.remove(os.path.join(root, file))
                print(f"Deleted: {os.path.join(root, file)}")

def debug_image(dataset_path) -> None:
    image = tf.io.read_file(dataset_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    
    plt.imshow(image.numpy().astype("uint8"))  # Convert to uint8 before displaying
    plt.show()

def setup_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    # Define train-validation split
    train_size = 0.8  # 80% training, 20% validation
    val_size = 1 - train_size

    # Split dataset
    #train_dataset = train_dataset.take(train_batches)

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH,
        image_size=IMAGE_SIZE,
        batch_size=64,
        validation_split=0.2,  # Use 20% for validation
        subset="training",
        seed=123,  # Ensures consistent split
        interpolation="nearest"  
    ) 
            
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))

        

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH,
        image_size=IMAGE_SIZE,
        batch_size=64,
        validation_split=0.2,  # Use same split
        subset="validation",
        seed=123,
        interpolation="nearest" 
    )
    #val_dataset = train_dataset.skip(train_batches)
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))



    # Get total number of batches
    total_batches = len(train_dataset)

    # Calculate number of training batches
    train_batches = int(total_batches * train_size)
    
    return train_dataset, val_dataset

# Build CNN model

def create_model() -> tf.keras.Model:

    model = keras.Sequential([
            keras.Input(shape=(64, 64, 3)),  # Define input explicitly
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),  # Reduce size
            layers.Conv2D(32, (3, 3), activation='relu'),

            layers.GlobalAveragePooling2D(),
            #layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),  # Helps prevent overfitting
            layers.Dense(2, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()
    
    return model

def train_evaluate_save_model(model, train_dataset, val_dataset) -> None:
    # Train the model
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=20)

    # Evaluate the model
    val_loss, val_acc = model.evaluate(val_dataset)
    print(f"Validation Accuracy: {val_acc:.2f}")

    model.save("tom_and_jerry_classifier.h5")


def predict_images_in_directory(directory_path, model):
    class_names = ["jerry","tom"]  # Update class labels
    predictions = {}  # Store predictions for each image

    accurate_toms = 0
    total_toms = 0
    accurate_jerrys = 0
    total_jerrys = 0
    # Loop through all image files in the directory
    for subdirectory in os.listdir(directory_path):
     
    # Ensure it's a directory before listing files            
        subdirectory_path = os.path.join(directory_path, subdirectory)
        for filename in os.listdir(subdirectory_path):
            file_path = os.path.join(subdirectory_path, filename)

            try:
                # Load and preprocess image
                img = tf.keras.preprocessing.image.load_img(file_path, target_size=(64, 64))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                img_array /= 255.0  # Normalize

                # Predict class
                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction)  # Get index of highest probability

                # Store result
                predictions[filename] = class_names[predicted_class]
                #print(f"{subdirectory_path}/{filename}({subdirectory}): {predictions[filename]}")
                # Count accurate predictions
                if (subdirectory == "tom"):
                    total_toms += 1
                    if (predictions[filename] == "tom"):
                        accurate_toms += 1
                    
                elif (subdirectory == "jerry"):
                    total_jerrys += 1
                    if (predictions[filename] == "jerry"):
                        accurate_jerrys += 1
                        

            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")
    print("Accurate toms: ", accurate_toms, "Accurate jerrys: ", accurate_jerrys)
    print("Total toms: ", total_toms, "Total jerrys: ", total_jerrys)
    print("Tom accuracy ", accurate_toms/total_toms, "Jerry accuracy: ", accurate_jerrys/total_jerrys)
    print("Total accuracy ", (accurate_toms/total_toms + accurate_jerrys/total_jerrys) / 2)


    
    return predictions  # Return all results as a dictionary

def load_model() -> tf.keras.Model:
    return keras.models.load_model("tom_and_jerry_classifier.h5")

def predict_single_image(image_path, model) -> None:
    class_names = ["jerry", "tom"]  # Same class label order used during training

    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        print(f"Prediction for {image_path}: {class_names[predicted_class]}")
        return class_names[predicted_class]

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def try_single_images() -> None:
    loaded_model = load_model()
    for filename in os.listdir(SINGLE_IMAGE_TESTS_PATH):
        file_path = os.path.join(SINGLE_IMAGE_TESTS_PATH, filename)
        predict_single_image(file_path, loaded_model)




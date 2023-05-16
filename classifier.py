
#Base Imports
from dataclasses import dataclass
from distutils.command.config import config
import sys
from grpc import Call
import numpy as np
import PIL
import tensorflow as tf
import pathlib
import os

#TF and Keras imports
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.callbacks import Callback

#PyDantic Imports
from pydantic import BaseModel
from typing import List

# Config for training of a new model
class TrainingConfig(BaseModel):
    epochs: int | None = None
    batchSize: int | None = None
    mode: str | None = None
    trainingSplit: float | None = None
    validationSplit: float | None = None

# Datatype for importing a Class of Images
class ImageClass(BaseModel):
    name: str
    images: List[PIL.Image.Image]

    class Config:
        arbitrary_types_allowed = True

# Datatype for importing a Dataset
class ImageDataset(BaseModel):
    name: str
    classes: List[ImageClass]


class ClassifierWrapper:
    def __init__(self, basepath="."):
        self.img_height = 180
        self.img_width = 180
        self.basepath = basepath
        return
    
    def save(self, modelName: str):
        self.model.save(f'{self.basepath}/models/{modelName}')

    def load(self, modelName: str):
        try:
            # Load the model from file
            # And get the classnames from
            # the model's dataset
            self.model = keras.models.load_model(f'{self.basepath}/models/{modelName}')
            self.class_names = self.getClassNames(modelName)

            return True
        except Exception as e:
            print(e)
            return False

    # Train dataset, save model, and quit.
    # Designed to be used in a separate thread
    # to be self closing
    def createAndSaveBG(self, modelName: str, config: TrainingConfig, callbacks: Callback|None = None):
        self.new(modelName, config, callbacks)
        self.save(modelName)
        sys.exit()

    def new(self, modelName: str, config: TrainingConfig, callbacks: Callback|None = None):
        options = config.dict(exclude_none=True)
        mode = options.get('mode', 'classic')

        if mode == 'classic':
            self.train(modelName, options, callbacks)
        elif mode == 'oneshot':
            self.trainOneshot(modelName, options, callbacks)
        else:
            self.train(modelName, options, callbacks)

        return

    def predict(self, image):
        image = image.resize((self.img_height, self.img_width)) # Different from the guide, may cause problems

        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        return {
            "prediction": self.class_names[np.argmax(score)],
            "confidence": 100 * np.max(score)
        }

    # --- Utility Methods ---
    def getClassNames(self, name: str):
        labels = []

        try:
            for root, dirs, files in os.walk(f'{self.basepath}/datasets/{name}/'):
                label = root.split('/')[4]
                if(label != ''):
                    labels.append(label)
        except:
            pass

        return labels

    def getClassData(self, name: str, classname: str):
        try:
            for root, dirs, files in os.walk(f'{self.basepath}/datasets/{name}/{classname}'):
                return files
        except:
            return []

    def hasDataset(self, dataset: str):
        return os.path.exists(f'{self.basepath}/datasets/{dataset}/')

    def hasClass(self, dataset: str, classname: str):
        return os.path.exists(f'{self.basepath}/datasets/{dataset}/{classname}/')

    def hasClassData(self, dataset: str, classname: str, classdata: str):
        return os.path.exists(f"{self.basepath}/datasets/{dataset}/{classname}/{classdata}")

    def importDataset(self, dataset: ImageDataset):
        if(not self.hasDataset(dataset.name)):
            os.mkdir(f'{self.basepath}/datasets/{dataset.name}/')
        
        for imageClass in dataset.classes:
            if(not self.hasClass(dataset.name, imageClass.name)):
                os.mkdir(f'{self.basepath}/datasets/{dataset.name}/{imageClass.name}/')
            
            i = 0
            for image in imageClass.images:
                temp = i
                while os.path.exists(f'{self.basepath}/datasets/{dataset.name}/{imageClass.name}/image-{temp}.jpg'):
                    temp += 1           
                
                image.save(f'{self.basepath}/datasets/{dataset.name}/{imageClass.name}/image-{temp}.jpg')       
        
        return

    # --- Training Functions ---
    def trainOneshot(self, modelName: str, options):
        print('DEBUG')
        
        data_dir = pathlib.Path(f'{self.basepath}/datasets/{modelName}')

        # Initialize training dataset
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=options.get('batchSize', 32)
        )

        # Initialize validation dataset
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=options.get('batchSize', 32)
        )

        # Store the class names from the directory
        # structure
        self.class_names = train_ds.class_names

        AUTOTUNE = tf.data.AUTOTUNE

        # Cache the Dataset
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Augment the images to prevent over-fitting
        data_augmentation = keras.Sequential([
            layers.RandomFlip(
                "horizontal",
                input_shape=(
                    self.img_height,
                    self.img_width,
                    3
                )
            ),
            layers.RandomRotation(0.1),
            layers.RandomRotation(0.2),
            layers.RandomRotation(0.3),
            layers.RandomRotation(0.4),
            layers.RandomRotation(0.5),
            layers.RandomRotation(0.6),
            layers.RandomRotation(0.7),
            layers.RandomRotation(0.8),
            layers.RandomRotation(0.9),
            layers.RandomZoom(0.1),
            layers.RandomZoom(0.2),
            layers.RandomZoom(0.3),
            layers.RandomZoom(0.4),
            layers.RandomZoom(0.5),
            layers.RandomZoom(0.6),
            layers.RandomZoom(0.7),
            layers.RandomZoom(0.8),
            layers.RandomZoom(0.9),
        ])

        # Define the model's layers
        self.model = Sequential([
            data_augmentation,
            layers.Rescaling(1./255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.class_names))
        ])

        # Compile the model's layers
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Train the model
        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=options.get('epochs', 10)
        )

        return

    def train(self, modelName: str, options: dict, callbacks: Callback|None = None):
        data_dir = pathlib.Path(f'{self.basepath}/datasets/{modelName}')

        # Only prep callbacks if provided
        callbackList = []
        if(not callbacks == None):
            callbackList.append(callbacks)

        # Initialize training dataset
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=options.get('trainingSplit',0.2),
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=options.get('batchSize', 32)
        )

        # Initialize validation dataset
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=options.get('validationSplit',0.2),
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=options.get('batchSize', 32)
        )

        # Store the class names from the directory
        # structure
        self.class_names = train_ds.class_names

        AUTOTUNE = tf.data.AUTOTUNE

        # Cache our dataset
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Apply augmentation layers for variance
        data_augmentation = keras.Sequential([
            layers.RandomFlip(
                "horizontal",
                input_shape=(
                    self.img_height,
                    self.img_width,
                    3
                )
            ),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

        # Normalize layers for training
        self.model = Sequential([
            data_augmentation,
            layers.Rescaling(1./255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.class_names))
        ])

        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Train the model on our data
        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=options.get('epochs', 10),
            callbacks=callbackList
        )

        return
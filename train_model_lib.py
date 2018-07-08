import os
import numpy as np
import pandas as pd
import cv2
import json

from keras.models import Sequential
from keras.layers import Convolution2D, Activation, ELU, Dropout, MaxPooling2D, Flatten, Dense, Lambda
from keras.preprocessing.image import img_to_array, load_img
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import model_from_json

UDACITY_TRAINING_DATA_DIR = "./udacity_data/"
UDACITY_TRAINING_DATA_CSV = "driving_log.csv"

# Custom model target shape
IMAGE_TARGET_SHAPE = (96,96)

# Comma AI Target shape
#IMAGE_TARGET_SHAPE = (128,128)


def load_data_from_file(data_file_dir, data_file_name):
    """
    Load data from csv lof file.
    Some data cleaning procedure are performed as adding file data dir to images paths and white spaces removal.
    """
    data_df = pd.read_csv(data_file_dir+data_file_name,names=None, usecols=[0, 1, 2, 3])
    
    #print(data_df.head())
    
    if data_file_dir == UDACITY_TRAINING_DATA_DIR:
        data_df["center"] = data_df["center"].apply(lambda x: data_file_dir + x)
        data_df["left"] = data_df["left"].apply(lambda x: data_file_dir + x)
        data_df["right"] = data_df["right"].apply(lambda x: data_file_dir + x)

    # Remove spaces from file paths
    data_df["center"] = data_df["center"].apply(lambda x: x.replace(" ",""))
    data_df["left"] = data_df["left"].apply(lambda x: x.replace(" ",""))
    data_df["right"] = data_df["right"].apply(lambda x: x.replace(" ",""))
        
    data_df["steering"] = data_df["steering"].astype(np.float32)

    #print(data_df.head())
    
    return data_df

def shuffle_data(data_df):
    """
    Performs data shuffling.
    """
    return data_df.sample(frac=1).reset_index(drop=True)

def split_data(data_df, split_ratio):
    """
    Split data into training and validation sets.
    """
    data_df_rows = int(data_df.shape[0] * split_ratio)

    training_data = data_df.loc[0:data_df_rows -1]
    validation_data = data_df.loc[data_df_rows:]

    return training_data, validation_data

def vertical_crop(image):
    """
    Remove first 50 rows from the images to discard the orizon.
    Remove also the last 30 rows to discard the car's bonnet.
    """
    # Horizon
    upper_crop_treshold = 50
    # Bonnet
    lower_crop_threshold = 130
    cropped_image = image[upper_crop_treshold:lower_crop_threshold,:,:]
    
    return cropped_image

def vertical_crop_sim(image):
    """
    Function used only to show the crop effect on a original image.
    First 50 rows and last 30 are made black.
    """    
    # Horizon
    upper_crop_treshold = 50
    # Bonnet
    lower_crop_threshold = 130

    cropped_image = image
    cropped_image[:upper_crop_treshold,:,:] = 0
    cropped_image[lower_crop_threshold:,:,:] = 0
    
    return cropped_image

def resize_to_target_size(image):
    """
    Image resized accordingly with IMAGE_TARGET_SHAPE touple
    """
    return cv2.resize(image, IMAGE_TARGET_SHAPE)

def normalize_image(image):
    """
    Image values normalized between -1 and +1.
    """
    a = -1.0
    b = 1.0
    px_min = 0
    px_max = 255
    normalized_image =  a + ( ( (image - px_min)*(b - a) )/( px_max - px_min ) )
   
    return normalized_image

def preprocess_image(image):
    """
    Image preprocessing pipeline used by training, validation and test data.
    """    
    # Crop
    image = vertical_crop(image)
    # Resize
    image = resize_to_target_size(image)
    # Float conversion
    image = image.astype(np.float32)
    #Normalize
    image = normalize_image(image)
    
    return image

def camera_chooser(row, debug=False):
    """
    Choose randomly witch camera image load [left, center, right]
    Adjust the steering angle accordingly
    If left is choosed add +0.2
    If right is choosed add -0.2
    """
    side = np.random.choice(["left", "center", "right"])
    steering = row["steering"]
    
    if debug:
        print("Original steering: {}".format(steering))
        print("Chosen side: {}".format(side))
    
    if side == "right":
        steering -= 0.2
    elif side == "left":
        steering += 0.2
    
    if debug:
        print("Final steering: {}".format(steering))
    
    image = load_img(row[side].strip())
    image = img_to_array(image)
    
    return image, steering

def horizontal_flip(image, steering, debug=False):
    """
    Choose randomly to split images horizontally.
    If split occurs steering angle sings will be inverted.
    """
    
    if debug:
        print("Preflip steering: {}".format(steering))        
    
    if np.random.randint(2) == 1:
        image = cv2.flip(image, 1)
        steering = steering * -1
        
    if debug:
        print("After flip steering: {}".format(steering))  
      
    return image, steering 

def change_image_brightness(image_rgb, debug=False):
    """
    Change image brightness randomly.
    """
    # Conver image from RGB to HSV (hue saturation value)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    brightness_mul = 1.0 + 0.5*(2*np.random.uniform()-1.0)
    
    if debug:
        print("Brightness mult factor {}".format(brightness_mul))
        
    # Apply transformation
    image_hsv[:,:,2] = image_hsv[:,:,2] * brightness_mul
    
    # Conver image from HSV (hue saturation value) to RGB
    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    
    return image_rgb

def _augment_row(row):
    """
    Data augmentation pipeline used by training and validation data.
    """
    steering = row['steering']

    image, steering = camera_chooser(row)    
    image, steering = horizontal_flip(image, steering)
    image = change_image_brightness(image)

    # Crop, resize and normalize the image
    image = preprocess_image(image)
    return image, steering

def _data_generator(df, batch_size=32):
    """
    Generator function used by training and validation data.
    """
    
    batch_to_generate = df.shape[0] // batch_size

    batch_counter = 0
    while(True):
        start_index = batch_counter * batch_size
        end_index  = start_index + batch_size - 1

        X_augm_batch = np.zeros((batch_size, IMAGE_TARGET_SHAPE[0], IMAGE_TARGET_SHAPE[1], 3), dtype=np.float32)        
        y_augm_batch = np.zeros((batch_size,), dtype=np.float32)

        # Extract a chunk of data and apply augmentation
        batch_row_index = 0
        for index, df_row in df.loc[start_index:end_index].iterrows():
            # Data augmentation and preprocessing
            X_augm_batch[batch_row_index], y_augm_batch[batch_row_index] = _augment_row(df_row)
            batch_row_index = batch_row_index + 1

        batch_counter = batch_counter + 1
        if batch_counter == batch_to_generate - 1:
            # index reset
            batch_counter = 0
        yield X_augm_batch, y_augm_batch


def get_model_to_train_custom():
    """
    Custom model developed for the project
    """    
    model = Sequential()
    
    # Image normalization with a Lambda function
    # model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(IMAGE_TARGET_SHAPE[0], IMAGE_TARGET_SHAPE[1], 3)))
    
    # Convolution
    model.add(Convolution2D(32, 3, 3, input_shape=(IMAGE_TARGET_SHAPE[0], IMAGE_TARGET_SHAPE[1], 3), border_mode='same', subsample=(2, 2), name="Conv2d_1"))
    model.add(Activation('relu', name="ReLU_1"))

    # Max pooling
    model.add(MaxPooling2D(pool_size=(2, 2), name="MAxPool_1"))

    # Convolution
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), name="Conv2d_2"))
    model.add(Activation('relu', name="ReLU_2"))

    # Max pooling
    model.add(MaxPooling2D(pool_size=(2, 2), name="MaxPool_2"))
    
    # Convolution
    model.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(1, 1), name="Conv2d_3"))
    model.add(Activation('relu', name="ReLU_3"))

    model.add(Flatten(name="Flatten_4"))
    model.add(Dropout(0.5, name="Dropout_4"))

    # Fully connected layer
    model.add(Dense(128, name="Dense_5"))
    model.add(Activation('relu', name="ReLU_5"))
    model.add(Dropout(0.5, name="Dropout_5"))
    
    # Fully connected layer   
    model.add(Dense(128, name="Dense_6"))
    
    # Fully connected layer
    model.add(Dense(1, name="Dense_7"))
    
    model.summary()
    
    adam = Adam(lr=0.0001)
    
    
    model.compile(optimizer=adam, loss='mse')
    
    return model

def get_model_to_train_commaai():
    """
    Creates the comma.ai model, and returns a reference to the model
    The comma.ai model's original source code is available at:
    https://github.com/commaai/research/blob/master/train_steering_model.py
    """
    #ch, row, col = CH, H, W  # camera format
    #H, W, CH = 160, 320, 3
    model = Sequential()
    
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), input_shape=(IMAGE_TARGET_SHAPE[0], IMAGE_TARGET_SHAPE[1], 3),border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    
    model.summary()    
    
    model.compile(optimizer="adam", loss='mse')
    
    return model 

def get_model_to_train():
    
    #return get_model_to_train_commaai()
    return get_model_to_train_custom()
    
def save_model_and_weights(model, model_weights_f, model_json_f):
    """
    Save model json and model weights to filesystem.
    """
    try:
        os.remove(model_json_f)
        os.remove(model_weights_f)
    except OSError:
        pass   

    with open(model_json_f, 'w') as outfile:
        json.dump(model.to_json(), outfile)
    model.save_weights(model_weights_f)
    
    print("Saved model json: {}".format(model_json_f))
    print("Saved model weights: {}".format(model_weights_f))

def inference_preprocessing(image):
    """
    Preprocessing stage for images during model execution (drive.py)
    No data augmentation.    
    """
    image = preprocess_image(image)
    
    return image
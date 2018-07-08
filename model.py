# Added custom function import
import train_model_lib as tm_lib

# Fix error with Keras and TensorFlow
import tensorflow as tf
#.control_flow_ops = tf

# Training data configuration
TRAINING_DATA_DIR = "./udacity_data/"
TRAINING_DATA_CSV = "driving_log.csv"

# Model saving configuration
MODEL_FILE = "models/model_custom_96x96_second_submit_relu.h5"


# Model training configuration
TRAINING_VALIDATION_SPILT = 0.8
BATCH_SIZE = 32
SAMPLES_PER_EPOCH = 25000
NB_EPOCHS = 5
NB_VAL_SAMPLES = 5000

if __name__ == '__main__':

    # Load data from file
    recorded_data_df = tm_lib.load_data_from_file(TRAINING_DATA_DIR, TRAINING_DATA_CSV)

    # Shuffle data
    recorded_data_df = tm_lib.shuffle_data(recorded_data_df)

    # Split training set and validation set
    training_data, validation_data =  tm_lib.split_data(recorded_data_df, TRAINING_VALIDATION_SPILT)

    # Print data sets shapes
    print("Training set shape {}".format(training_data.shape))
    print("Validation set shape {}".format(validation_data.shape))

    # Release main variable to free memory
    recorded_data_df = None

    # Setup data generators
    training_data_generator = tm_lib._data_generator(training_data, batch_size=BATCH_SIZE)
    validation_data_generator = tm_lib._data_generator(validation_data, batch_size=BATCH_SIZE)

    # Get NN Model
    model = tm_lib.get_model_to_train()

    # Train model
    samples_per_epoch = (SAMPLES_PER_EPOCH//BATCH_SIZE)*BATCH_SIZE

    model.fit_generator(
        training_data_generator,
        validation_data=validation_data_generator,
        samples_per_epoch=samples_per_epoch,
        nb_epoch=NB_EPOCHS,
        nb_val_samples=NB_VAL_SAMPLES
    )

    # Save model
    model.save(MODEL_FILE)
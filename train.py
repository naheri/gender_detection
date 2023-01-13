from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.models import load_model
from build import create_model
from keras.utils import plot_model

# config
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
# tensorflow version
print('Tensorflow version:', tf.__version__)

# directory paths
train_dir = 'dataset/Train'
test_dir = 'dataset/Test'
val2_dir = 'dataset/Validation'

# HyperParameters
TARGET_SIZE = (64, 64)
BATCH_SIZE = 32
CLASS_MODE = 'binary'  # 2 classes (0, 1)
learning_rate = 1e-3
epochs = 15

# Data Augmentation
aug_train = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
aug_test = ImageDataGenerator(
    rescale=1./255,
)

# Connecting the ImageDataGenerator objects to our dataset

train_generator = aug_train.flow_from_directory(train_dir,
                                                 target_size=TARGET_SIZE,
                                                 subset='training',
                                                 batch_size=BATCH_SIZE,
                                                 class_mode=CLASS_MODE)

validation_generator = aug_train.flow_from_directory(train_dir,
                                                      target_size=TARGET_SIZE,
                                                      subset='validation',
                                                      batch_size=BATCH_SIZE,
                                                      class_mode=CLASS_MODE)
# small test set : 200 images
test_generator = aug_test.flow_from_directory(test_dir,
                                               target_size=TARGET_SIZE,
                                               batch_size=BATCH_SIZE,
                                               class_mode=CLASS_MODE)
# large test set : 11649 images
val2_generator = aug_test.flow_from_directory(val2_dir,
                                               target_size=TARGET_SIZE,
                                               batch_size=BATCH_SIZE,
                                               class_mode=CLASS_MODE)

num_of_train_samples = train_generator.n
num_of_val_samples = validation_generator.n
num_of_test_samples = test_generator.n

# Model
model = create_model(width=TARGET_SIZE[0], height=TARGET_SIZE[1], depth=3, learning_rate=learning_rate, epochs=epochs)
# Model train
H = model.fit(train_generator,
        epochs = epochs,
        verbose = 1,
        validation_data = validation_generator,
        # save the models through the training
         callbacks= [ModelCheckpoint('saved_models/model_{val_accuracy:.3f}.h5',
                                           save_best_only=True,
                                           save_weights_only=False,
                                           monitor='val_accuracy',
                                             ), 
                    EarlyStopping(monitor='val_accuracy', patience=5) # stop training if the val_accuracy doesn't improve after 5 epochs
                    ])
        
        

# Load model
model = load_model('models1/model_0.968.h5') # best model

# model architecture
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)







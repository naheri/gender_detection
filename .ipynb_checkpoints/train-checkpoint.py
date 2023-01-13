from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import seaborn as sns
from keras.utils import plot_model
from build import create_model

# hyperparameters
target_size = (64, 64)
batch_size = 32
class_mode = 'binary'  
num_of_train_samples = 1500
num_of_val_samples = 400

target_names = ['female','male']

# data directories
train_dir = 'dataset/Training'
val_dir = 'dataset/Validation'

# Creating ImageDataGenerator objects
dgen_train = ImageDataGenerator(rescale=1./255,
                                rotation_range=25,
                                validation_split=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)
dgen_test = ImageDataGenerator(rescale=1./255)


# Connecting the ImageDataGenerator objects to our dataset
train_generator = dgen_train.flow_from_directory(train_dir,
                                                 target_size=target_size,
                                                 batch_size=batch_size,
                                                 class_mode=class_mode)

validation_generator = dgen_test.flow_from_directory(val_dir,
                                                      target_size=target_size,
                                                      batch_size=batch_size,
                                                      class_mode=class_mode)

# Get the class indices
print(train_generator.class_indices)

# Get the image shape
print(train_generator.image_shape)

# initialize the model
model = create_model(width=target_size[0], height=target_size[1], depth=3)
# Train the Model
trained_model = model.fit(train_generator,
        steps_per_epoch=num_of_train_samples // batch_size,
        validation_steps=num_of_val_samples // batch_size,
        epochs=25,
        validation_data=validation_generator,
        callbacks=[
        # Stopping our training if val_accuracy doesn't improve after 20 epochs
        EarlyStopping(monitor='val_accuracy', 
                                        patience=20),
        # Saving  of our model in the model directory
    
        # save the best weights and the model architecture
        tf.keras.callbacks.ModelCheckpoint('models/model_{val_accuracy:.3f}.h5',
                                        save_best_only=True,
                                        save_weights_only=False,
                                           monitor='val_accuracy',
                                             )
    ])

# model architecture
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)



#Confusion Matrix and Classification Report
Y_pred = model.predict(validation_generator, num_of_val_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')

print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
cm = confusion_matrix(validation_generator, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

disp.plot(cmap=plt.cm.Blues)
plt.show()


# Plot the model, initialize figure and axes
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Plot training and validation loss
axs[0].plot(trained_model.trained_model['loss'])
axs[0].plot(trained_model.trained_model['val_loss'])
axs[0].legend(['Training', 'Validation'])
axs[0].set_title('Training and Validation Losses')
axs[0].set_xlabel('epochs')

# Plot training and validation accuracy
axs[1].plot(trained_model.trained_model['accuracy'])
axs[1].plot(trained_model.trained_model['val_accuracy'])
axs[1].legend(['Training', 'Validation'])
axs[0].set_title('Training and Validation Accuracy')
axs[1].set_xlabel('epochs')

plt.show()







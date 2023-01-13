# Importing Libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import FalsePositives,FalseNegatives,BinaryAccuracy,TrueNegatives,TruePositives



def create_model(height, width, depth, learning_rate, epochs) -> Sequential(): 
    model = Sequential()
    # for BatchNormalization
    inputShape = (height, width, depth)
    # Building CNN Model
    model = Sequential()
    model.add(Conv2D(32, (5,5), padding='same', activation='relu',
                    input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=learning_rate, decay = learning_rate/epochs),
                loss=BinaryCrossentropy(),
                metrics=[BinaryAccuracy(),
                        FalseNegatives(),
                        FalsePositives(),
                        TrueNegatives(),
                        TruePositives(),
                        'accuracy',
    ])
    model.summary()
    


    return model
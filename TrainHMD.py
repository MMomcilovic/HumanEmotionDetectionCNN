
# import required packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Initialize image data generator with rescaling
from keras.utils import plot_model
from matplotlib import pyplot as plt

train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

ferTest = 'data/FER/test'
ferTrain = 'data/FER/train'
ckTest = 'data/CK+48/test'
ckTrain = 'data/CK+48/train'

# Preprocess all test images
train_generator = train_data_gen.flow_from_directory(
        ferTrain,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
        ferTest,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# create model structure
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

plot_model(emotion_model, to_file='model2.png')

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# Train model
emotion_model_info = emotion_model.fit(
        train_generator,
        steps_per_epoch=400,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=100)

# emotion_model_info = emotion_model.fit(
#         train_generator,
#         steps_per_epoch=20,
#         epochs=10,
#         validation_data=validation_generator,
#         validation_steps=5)

# save model structure in jason file
model_json = emotion_model.to_json()
with open("./model/emotion_model_v3.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion_model.save_weights('./model/emotion_model_v3.h5')

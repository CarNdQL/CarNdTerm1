import csv
import cv2
import numpy as np

import sklearn


# define generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images, angles = [],[]
            # batch_sample structure:[filename, angle, flag_to_flip]
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample[0])
                angle = batch_sample[1]
                if  batch_sample[2]==1:
                    image = cv2.flip(image,1)
                    angle = angle*-1.0
                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# load driving log data to samples
samples = []
angle_correction = [0, 0.2, -0.2] # used to estimate the angle of left/right images

# folder "sim_training" contains the recording of two laps on track one using center lane driving.
input_path = 'sim_training/'
with open(input_path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        angle = float(line[3])
        # generated 6 samples from each line:
        #  center, center_flipped, left, left_flipped, right, right_flipped
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = input_path+'IMG/'+filename
            # sample structure:[filename, angle, flag_to_flip]
            sample=[current_path, angle+angle_correction[i], 0]
            samples.append(sample)
            sample=[current_path, angle+angle_correction[i], 1]
            samples.append(sample)

# folder "sim_training_more" contains the recording of recovering from the left side and right sides of the road back to center.
input_path = 'sim_training_more/'
with open(input_path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        angle = float(line[3])
        for i in range(1):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = input_path+'IMG/'+filename
            # sample structure:[filename, angle, flag_to_flip]
            sample=[current_path, angle+angle_correction[i], 0]
            samples.append(sample)
            sample=[current_path, angle+angle_correction[i], 1]
            samples.append(sample)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# lambda layer:  image normalization
model.add(Lambda(lambda x:x/255.0-0.5, input_shape=(160,320,3)))
# cropping layer: only keep the portion of the image that is useful for predicting the steering angle.
model.add(Cropping2D(cropping=((70,25),(0,0))))
# NVIDIA Architecture
# Convolution2D layer 1-5
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.1))
# Fully connected layer 1-4
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# use mean-square-error and adam-optimizer to train the model for regression
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=
            len(train_samples), validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

model.save('model.h5')


import matplotlib.pyplot as plt
# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

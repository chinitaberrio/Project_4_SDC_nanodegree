import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn

lines = []
with open('new_data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    # loops forever
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample[0])
                images.append(image)
                images.append(cv2.flip(image, 1))
                angle = float(batch_sample[3])
                angles.append(angle)
                angles.append(angle * -1.0)
            x_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(x_train, y_train)


# batch size
batch_size = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
valid_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Creating the model
model = Sequential()
# Normalizing the data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# Cropping non relevant information
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_obj = model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples) / batch_size),
                                  validation_data=valid_generator,
                                  validation_steps=np.ceil(len(validation_samples) / batch_size),
                                  epochs=7, verbose=1)

print(history_obj.history.keys())

# plotting the history of the training
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
exit()

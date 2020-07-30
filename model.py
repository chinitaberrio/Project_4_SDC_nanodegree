import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn

lines = []
with open('data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_samples, test_samples = train_test_split(train_samples, test_size=0.1)

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
                correction = 0.218
                # centre image
                source_path = batch_sample[0]
                filename = source_path.split('/')[-1]
                current_path = 'data/IMG/' + filename
                image = cv2.imread(current_path)
                images.append(image)
                images.append(cv2.flip(image, 1))
                angle = float(batch_sample[3])
                angles.append(angle)
                angles.append(angle * -1.0)
                # left image
                source_path = batch_sample[1]
                filename = source_path.split('/')[-1]
                current_path = 'data/IMG/' + filename
                image = cv2.imread(current_path)
                images.append(image)
                angles.append(angle+correction)
                # right image
                source_path = batch_sample[2]
                filename = source_path.split('/')[-1]
                current_path = 'data/IMG/' + filename
                image = cv2.imread(current_path)
                images.append(image)
                angles.append(angle - correction)

            x_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(x_train, y_train)


# batch size, half due to data augmentation
batch_size = 12

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
valid_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

# Creating the model
model = Sequential()
# Normalizing the data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# Cropping non relevant information
model.add(Cropping2D(cropping=((60, 20), (0, 0))))
# First Convolutional layer
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
# Second Convolutional layer
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
# Dropout of 30% of the connections
model.add(Dropout(0.3))
# Fourth Convolutional layer
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
# Fifth Convolutional layer
model.add(Convolution2D(64, 3, 3, activation="relu"))
# Dropout of 30% of the connections
model.add(Dropout(0.3))
# Sixth Convolutional layer
model.add(Convolution2D(64, 3, 3, activation="relu"))
# Flatten
model.add(Flatten())
# First Fully conected layer
model.add(Dense(100))
# Second Fully conected layer
model.add(Dense(50))
# Output layer
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_obj = model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples) / batch_size),
                                  validation_data=valid_generator,
                                  validation_steps=np.ceil(len(validation_samples) / batch_size),
                                  epochs=4, verbose=1)

# Saving the model
model.save('model.h5')

# testing the model
images_test = []
angles_test = []
for sample in test_samples:
    source_path = sample[0]
    filename = source_path.split('/')[-1]
    current_path = 'new_data/IMG/' + filename
    image = cv2.imread(current_path)
    images_test.append(image)
    angle = float(sample[3])
    angles_test.append(angle)
images_test = np.array(images_test)
angles_test = np.array(angles_test)

# plotting the history of the training
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

metrics = model.evaluate(images_test, angles_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))

exit()

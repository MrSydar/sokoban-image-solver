from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np
from PIL import Image
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

model_export_path = "C:/Users/Den/PycharmProjects/sokoban-image-processing/tfmodels/training_2/cp-{epoch:04d}.ckpt"

train_dir = "C:/Users/Den/PycharmProjects/sokoban-image-processing/images/samples/training-symbols/"
train_players = train_dir + 'players/'
train_crosses = train_dir + 'crosses/'
train_esquares = train_dir + 'empty-squares/'
train_fsquares = train_dir + 'filled-squares/'

batch_size = 16
epochs = 15
IMG_HEIGHT = 50
IMG_WIDTH = 50

class_names = ['player', 'cross', 'esquare', 'fsquare']
train_labels = []
file_num = 0

files = os.listdir(train_players)
file_num += len(files)
train_labels += [0] * len(files)
players = np.ndarray(shape=(len(files), 50, 50), dtype=int)
for file in range(len(files)):
    image = Image.open(train_players + files[file])
    players[file] = np.asarray(image)

files = os.listdir(train_crosses)
file_num += len(files)
train_labels += [1] * len(files)
crosses = np.ndarray(shape=(len(files), 50, 50), dtype=int)
for file in range(len(files)):
    image = Image.open(train_crosses + files[file])
    crosses[file] = np.asarray(image)

files = os.listdir(train_esquares)
file_num += len(files)
train_labels += [2] * len(files)
esquares = np.ndarray(shape=(len(files), 50, 50), dtype=int)
for file in range(len(files)):
    image = Image.open(train_esquares + files[file])
    esquares[file] = np.asarray(image)

files = os.listdir(train_fsquares)
file_num += len(files)
train_labels += [3] * len(files)
fsquares = np.ndarray(shape=(len(files), 50, 50), dtype=int)
for file in range(len(files)):
    image = Image.open(train_fsquares + files[file])
    fsquares[file] = np.asarray(image)

train_images = np.concatenate((players, crosses, esquares, fsquares))

train_images = train_images / 255.0
# train_images[train_images > 230] = 1.0
# train_images[train_images <= 230] = 0.0

shuffler = np.random.permutation(len(train_images))
train_images = train_images[shuffler]
train_labels = np.asarray(train_labels)[shuffler]

model = Sequential([
    Flatten(input_shape=(50, 50)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])

# model = tensorflow.keras.models.load_model('tfmodels/model.h5')

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model.load_weights(model_export_path.format(epoch=0))
model.fit(train_images, train_labels, epochs=14, batch_size=32)

loss, acc = model.evaluate(train_images, train_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

if acc > 0.90:
    model.save_weights(model_export_path.format(epoch=0))
    model.save('tfmodels/model.h5')

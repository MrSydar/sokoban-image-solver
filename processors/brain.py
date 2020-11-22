from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow import argmax
import numpy as np


class Classifier:
    def recognise(self, img):
        img = img / 255.0
        img = np.expand_dims(img, 0)
        predarr = self.model.predict(img)[0]
        answer = np.argmax(predarr)
        return answer

    def __init__(self, weights_path):
        self.model = Sequential([
            Flatten(input_shape=(50, 50)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(4, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        self.model.load_weights(weights_path.format(epoch=0))

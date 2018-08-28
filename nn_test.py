
# import os
'''
with open(os.path.expanduser('~')+'\\.keras\\keras.json','w') as f:
    new_settings = """{\r\n
    "epsilon": 1e-07,\r\n
    "image_data_format": "channels_last",\n
    "backend": "theano",\r\n
    "floatx": "float32"\r\n
    }"""
    f.write(new_settings)
'''
import theano
# from keras_applications import *
from keras import backend as K
import os

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

np.random.seed(7)

# standard machine learning dataset from the UCI Machine Learning repository.

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

set_keras_backend("theano")


# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv",delimiter=",")

# split into input(X) and output(Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

first_layer = 12
second_layer = 8


# ===== create model =====
model = Sequential()
model.add(Dense(first_layer, input_dim=8, activation='relu'))
model.add(Dense(second_layer, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# ===== Compile Model =====
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ===== Fit the model =====
model.fit(X, Y, epochs=150, batch_size=10)

# ====== Evaluate Model ======
scores = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

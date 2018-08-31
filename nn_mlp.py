from keras import backend as K
import os

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.cross_validation import train_test_split
seed=7
np.random.seed(seed)


# Manually set Theano as the backend
def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

set_keras_backend("theano")

# load pima indians dataset
# standard machine learning dataset from the UCI Machine Learning repository.
dataset = np.loadtxt("pima-indians-diabetes.csv",delimiter=",")

# split into input(X) and output(Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# Split into 67% training and 33% test data set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

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
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=10)

# ====== Evaluate Model ======
scores = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# ============== Manual K Fold Cross Validation ====================

from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import StratifiedKFold
import numpy

# fix random seed for reproducing same output
seed = 10
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv",delimiter=",")

# split into input(X) and output(Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_folds=10,shuffle=True,random_state=seed)
cvscores = []

for train, test in kfold.test_folds(X,Y):

    model=Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    # Compile Model
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    # Fit the model
    model.fit(X[train],Y[train], epochs = 150, batch_size= 10, verbose=0)

    # Evaluate the model
    scores = model.evaluate(X[test],Y[test], verbose=0)

    print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))

    cvscores.append(scores[1]*100)

print('%.2f%% (+/- %.2f%%)' % (numpy.mean(cvscores), numpy.std(cvscores)))


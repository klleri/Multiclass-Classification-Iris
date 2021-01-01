
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score



rawData = pd.read_csv("iris.data")
rawData.info()

predictors = rawData.loc[:, 'sepal length':'petal width'].values
target  = rawData.loc[:, 'class'].values


labelencoder = LabelEncoder()
target = labelencoder.fit_transform(target)
targetBinary = np_utils.to_categorical(target)



def creatingNeuralNetwork():
    neuralNetwork = Sequential()
    neuralNetwork.add(Dense(units = 8, activation = 'softplus', kernel_initializer = 'random_normal', input_dim = 4))
    neuralNetwork.add(Dropout(0.1))
    neuralNetwork.add(Dense(units = 8, activation = 'softplus', kernel_initializer = 'random_normal'))
    neuralNetwork.add(Dropout(0.1))
    neuralNetwork.add(Dense(units = 8, activation = 'softplus', kernel_initializer = 'random_normal'))
    neuralNetwork.add(Dropout(0.1))
    neuralNetwork.add(Dense(units = 3, activation = 'softmax'))
    optimizer = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    neuralNetwork.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    return neuralNetwork

neuralNetwork = KerasClassifier(build_fn = creatingNeuralNetwork, epochs = 1000, batch_size = 10)
result = cross_val_score(estimator = neuralNetwork, X = predictors, y = target, cv = 10, scoring = 'accuracy')

#mean to calculate the mean of all Cross-Validation
mean = result.mean()
#if your standardDeviation return a high value, probabily your network is an overfitting network.
standardDeviation = result.std()
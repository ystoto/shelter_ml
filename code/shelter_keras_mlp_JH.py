import csv

import pandas as pd
import numpy as np

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler

def ageofyear(x):
    try:
        y = x.split()
    except:
        return None
    if 'year' in y[1]:
        return float(y[0]) * 365 * (1/365)
    elif 'month' in y[1]:
        return float(y[0]) * (365 / 12) * (1/365)
    elif 'week' in y[1]:
        return float(y[0]) * 7 * (1/365)
    elif 'day' in y[1]:
        return float(y[0]) * (1/365)


# 3) arrange function define
def arrange_train_data():

    print('Loading training data...')
    data = pd.read_csv('../input/train.csv')

    data['HasName'] = data['Name'].fillna(0)
    data.loc[data['HasName'] != 0,"HasName"] = 1
    data['HasName'] = data['HasName'].astype(int)
    data['AnimalType'] = data['AnimalType'].map({'Cat':0,'Dog':1})
    data['OutcomeType'] = data['OutcomeType'].map({'Return_to_owner': 1, 'Euthanasia': 2, 'Adoption': 3, 'Transfer': 4, 'Died': 0})

    gender = {'Neutered Male': 1, 'Spayed Female': 2, 'Intact Male': 3, 'Intact Female': 4, 'Unknown': 5, np.nan: 0}
    data['SexuponOutcome'] = data['SexuponOutcome'].map(gender)
    data['AgeInDays'] = data['AgeuponOutcome'].map(ageofyear)
    data.loc[(data['AgeInDays'].isnull()), 'AgeInDays'] = data['AgeInDays'].median()
    data['Name+Gender'] = data['HasName'] + data['SexuponOutcome']
    data['Type+Gender'] = data['AnimalType'] + data['SexuponOutcome']
    data['IsMix'] = data['Breed'].str.contains('mix', case=False).astype(int)

    result = data['OutcomeType']

    print("loading new_data ..")
    arranged_train_data = data.drop(['OutcomeType','AnimalID','OutcomeSubtype','AgeuponOutcome', 'Name', 'Breed', 'Color', 'DateTime'], axis=1)
    #print(arranged_train_data)

    return arranged_train_data, result

def arrange_test_data():

    print('Loading testing data...')
    data = pd.read_csv('../input/test.csv')

    data['HasName'] = data['Name'].fillna(0)
    data.loc[data['HasName'] != 0, "HasName"] = 1
    data['HasName'] = data['HasName'].astype(int)
    data['AnimalType'] = data['AnimalType'].map({'Cat': 0, 'Dog': 1})
    data['OutcomeType'] = data['OutcomeType'].map({'Return_to_owner': 1, 'Euthanasia': 2, 'Adoption': 3, 'Transfer': 4, 'Died':0})

    gender = {'Neutered Male': 1, 'Spayed Female': 2, 'Intact Male': 3, 'Intact Female': 4, 'Unknown': 5, np.nan: 0}
    data['SexuponOutcome'] = data['SexuponOutcome'].map(gender)
    data['AgeInDays'] = data['AgeuponOutcome'].map(ageofyear)
    data.loc[(data['AgeInDays'].isnull()), 'AgeInDays'] = data['AgeInDays'].median()

    data['Name+Gender'] = data['HasName'] + data['SexuponOutcome']
    data['Type+Gender'] = data['AnimalType'] + data['SexuponOutcome']
    data['IsMix'] = data['Breed'].str.contains('mix', case=False).astype(int)

    result = data['OutcomeType']

    print("loading new_data ..")
    arranged_test_data = data.drop(['OutcomeType', 'AnimalID', 'OutcomeSubtype', 'AgeuponOutcome', 'Name', 'Breed', 'Color', 'DateTime'], axis=1)

    return arranged_test_data, result


X_train, Y_train = arrange_train_data()
X_train = np.array(X_train, np.int32)
X_train = X_train.astype('float32')
#X_train /= 6375
print(X_train.shape[0], 'train samples')

X_test, Y_test = arrange_test_data()
X_test = np.array(X_test, np.int32)
X_test = X_test.astype('float32')
#X_test /= 6375
print(X_test.shape[0], 'test samples')

batch_size = 128
nb_classes = 5 # np.max(Y_train) + 1
nb_epoch = 20

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

model = Sequential()
model.add(Dense(512, input_shape=(len(X_train[0]),)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(5))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
print('---------------')
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

preds = model.predict_proba(X_test)

of = open('./prediction.csv', 'w')
csvWriter = csv.writer(of, delimiter=',')
csvWriter.writerow(['AnimalID', 'Adoption', 'Died', 'Euthanasia', 'Return_to_owner','Transfer'])
for i, row in enumerate(preds):
    csvWriter.writerow([i+1] + list(row))
of.close




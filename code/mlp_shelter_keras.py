'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
import pandas as pd
import csv

from keras.layers import ELU

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

### Configuration ###############
use_onehot = True
use_dropout = True
use_ELU = False
reuse_model_with_weight = False
batch_size = 128
nb_classes = 5
nb_epoch = 20
###################

def ageofyear(x):
    try:
        y = x.split()
    except:
        return None
    if len(y) <= 1:
        return 0
    elif 'year' in y[1]:
        return int(y[0]) * 365 / 90 + 300
    elif 'month' in y[1]:
        return int(y[0]) * (365 / 12) / 90 + 1000
    elif 'week' in y[1]:
        return int(y[0]) * 7 / 90 + 100# +100 to avoid being zero
    elif 'day' in y[1]:
        return int(y[0]) / 90 + 500 # +500 to avoid being zero

def split_slash(x):
    if ('Mix' in x):
        return 'Mix'
    elif ('/' in x):
        return x.split('/')[1]


def arrange_test_data(path, begin, end):
    print('Loading testing data...')
    data = pd.read_csv(path)
    result = pd.DataFrame()

    if end > 0:
        data = data.drop(data.index[end:])  ##  Remain 100 rows just for debugging

    if begin > 0:
        data = data.drop(data.index[:begin])  ##  Remain 100 rows just for debugging

    data.fillna('', inplace=True)
    ######## TODO: Adjust column ####################
    if (True):
        #data['IsMix'] = data['Breed'].str.contains('mix', case=False).astype(int)
        data['Breed2'] = data['Breed'].map(split_slash).fillna(value=0)
        data['Breed'] = data['Breed'].map(
            lambda x: ((x.split(' Mix')[0]).split('/')[0])
        )

        data['Color2'] = data['Color'].map(split_slash).fillna(value=0)
        data['Color'] = data['Color'].map(
            lambda x: (x.split('/')[0])
        )

        #data['MixColor'] = data['Color'].str.contains('/', case=False).fillna(value=0).astype(int)
        #data['Black'] = data['Color'].str.contains('Black', case=False).fillna(value=0).astype(int)
        #data['Red'] = data['Color'].str.contains('Red', case=False).fillna(value=0).astype(int)
        #data['Brown'] = data['Color'].str.contains('Brown', case=False).fillna(value=0).astype(int)

    data['IsNamed'] = data['Name'].map(
        lambda x: (len(x) > 0)
    )

    data['AgeInDays'] = data['AgeuponOutcome'].map(ageofyear).fillna(value=0)

    if (True):
        data['Month'] = data['DateTime'].map(
            lambda x: pd.tslib.Timestamp(x).month
        ).fillna(value=0)

        data['Year'] = data['DateTime'].map(
            lambda x: pd.tslib.Timestamp(x).year
        ).fillna(value=0)

    target_to_remove= ['ID']
    target_to_remove.append('Name')
    target_to_remove.append('DateTime')
    #target_to_remove.append('AnimalType')
    #target_to_remove.append('SexuponOutcome')
    target_to_remove.append('AgeuponOutcome')
    #target_to_remove.append('Breed')
    #target_to_remove.append('Color')
    ############################

    if 'OutcomeType' in data.columns:
        result = data['OutcomeType'].copy()
        target_to_remove.append('OutcomeType')
        target_to_remove.append('OutcomeSubtype')

    if 'AnimalID' in data.columns:
        target_to_remove.remove('ID')
        target_to_remove.append('AnimalID')
    #print(target_to_remove.



    arranged_test_data = data.drop(target_to_remove, axis=1)
    arranged_test_data = arranged_test_data.reset_index(drop=True)
    result = result.reset_index(drop=True)

    #print(arranged_test_data)
    #print(result)
    return arranged_test_data, result

def generateDict(dictionary, table):
    if not dictionary:
        dictionary = dict()

    for idx, row in table.iterrows():
        for columnHeader, col in row.iteritems():
            if (None == dictionary.get(columnHeader, None)):
                dictionary[columnHeader] = dict()
            if (None == dictionary[columnHeader].get(col, None)):
                dictionary[columnHeader][col] = len(dictionary[columnHeader]) #  0, 1, 2,,,
    print('--------------')
    for col in dictionary:
        print(col, " : ", len(dictionary[col]))
    print('--------------')
    return dictionary

def map_to_float(dictionary, table):
    for idx, row in table.iterrows():
        for columnHeader, col in row.iteritems():
            table.set_value(idx, columnHeader, dictionary[columnHeader][col])

    for columnHeader in table.columns:
        table[columnHeader] = table[columnHeader].astype(float)

    for idx, row in table.iterrows():
        for columnHeader, col in row.iteritems():
            table.set_value(idx, columnHeader, col / len(dictionary[columnHeader]))
    return np.array(table, np.float32)


def map_to_integer(dictionary, table):
    for idx, row in table.iterrows():
        for columnHeader, col in row.iteritems():
            table.set_value(idx, columnHeader, dictionary[columnHeader][col])#table[columnHeader][idx] = dictionary[columnHeader][col]
    return table

def map_to_integer_for_outcometype(table):
    dic = dict()
    for idx, val in table.iteritems():
        if None == dic.get(val, None):
            dic[val] = len(dic)
        table.set_value(idx, dic[val])
    return table

def to_categorical_2d(dictionary, table): ## We need dictionary to get max value of whole dataset (not only trainset but testset)
    ## 1. Find max of each column
    total_nb_classes = 0
    for idx, col in enumerate(dictionary):
        total_nb_classes += len(dictionary[col])
        #print('idx: ', idx, 'col: ', col, 'max: ',len(dictionary[col]), 'tot_max:', total_nb_classes)
    ## 2. generate np.zeros(len(table), sum(max))
    Y = np.zeros((len(table), total_nb_classes))
    print('table.shape: ', table.shape,' ---> NewTable.shape: ',Y.shape,',  len(dictionary): ', len(dictionary))

    ## 3. For all rows
    for idx, row in table.iterrows():
        ## 4.for all column
        new_col = 0
        for columnHeader, col in row.iteritems():
            ## 5.Insert data into new np array
            #print('row:', row, 'col: ', col, 'y[]: ', y[row, col], 'new_col: ', new_col)
            Y[idx, col +  new_col] = 1.
            new_col += len(dictionary[columnHeader])
    return Y


X_train, Y_train = arrange_test_data('../input/train.csv', 0, 0)
X_test, Y_test = arrange_test_data('../input/test.csv', 0, 0)  ## dummy for  dictionary generation
X_total = pd.concat([X_train, X_test])
print ('Merge train.csv and test.csv to get max number of data per each column: '\
       , len(X_total), ' = train.csv(', len(X_train), ') + test.csv(', len(X_test), ')')

dictionary = dict()
dictionary = generateDict(dictionary, X_total)
X_train, Y_train = arrange_test_data('../input/train.csv', 0, 27000)
X_test, Y_test = arrange_test_data('../input/train.csv', 25001, 27000)


####### TODO: Verify the data before one-hot
#print(dictionary)
print('final:', X_train.head(10))
########

if use_onehot:
    X_train = map_to_integer(dictionary, X_train)
    X_train = to_categorical_2d(dictionary, X_train)
    # TODO: Verify the data after one-hot
    X_train = np.array(X_train, np.int32)
else:
    X_train = map_to_float(dictionary, X_train)

Y_train = map_to_integer_for_outcometype(Y_train)
Y_train = np.array(Y_train, np.int32)
Y_train = np_utils.to_categorical(Y_train, nb_classes)

if use_onehot:
    X_test = map_to_integer(dictionary, X_test)
    X_test = to_categorical_2d(dictionary, X_test)
    X_test = np.array(X_test, np.int32)
else:
    X_test = map_to_float(dictionary, X_test)



Y_test = map_to_integer_for_outcometype(Y_test)
Y_test = np.array(Y_test, np.int32)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

'''
for i, row in enumerate(X_train[0:10]):
    print('X_train[',i,']: ', row[0:20])
print('X_train row: ', len(X_train), 'col: ', len(X_train[0]))
print('Y_train: ', Y_train[0:20], '...')
print('Y_train row: ', len(Y_train), 'col: ', len(Y_train[0]))

for i, row in enumerate(X_test[0:10]):
    print('X_test[',i,']: ', row[0:10])
print('X_test row: ', len(X_test), 'col: ', len(X_test[0]))
print('Y_test: ', Y_test[0:20], '...')
print('Y_test row: ', len(Y_test), 'col: ', len(Y_test[0]))
'''

if (reuse_model_with_weight):
    model = model_from_json(open('my_model_architecture.json').read())
    model.load_weights('my_model_weights.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
else:
    model = Sequential()

    model.add(Dense(2048, input_shape=(len(X_train[0]),)))
    if use_ELU:
        model.add(ELU(alpha=1.0))
    else:
        model.add(Activation('relu'))
    if use_dropout:
        model.add(Dropout(0.2))

    model.add(Dense(2048))
    if use_ELU:
        model.add(ELU(alpha=1.0))
    else:
        model.add(Activation('relu'))
    if use_dropout:
        model.add(Dropout(0.5))

    model.add(Dense(2048))
    if use_ELU:
        model.add(ELU(alpha=1.0))
    else:
        model.add(Activation('relu'))
    if use_dropout:
        model.add(Dropout(0.5))

    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    print('----start fit-----')
    history = model.fit(X_train, Y_train, shuffle=True,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, validation_split=0.2) # validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

if (reuse_model_with_weight):
    json_string = model.to_json()
    open('my_model_architecture.json', 'w').write(json_string)
    model.save_weights('my_model_weights.h5')



############ test

X_test, Y_test = arrange_test_data('../input/test.csv', 0, 0)
if use_onehot:
    print('final X_test:\n', X_test.head(10))
    X_test = map_to_integer(dictionary, X_test)
    print('final X_test:\n', X_test.head(10))
    X_test = to_categorical_2d(dictionary, X_test)
    X_test = np.array(X_test, np.int32)
else:
    X_test = map_to_float(dictionary, X_test)



print('before:\n', X_test[0], 'len: ', len(X_test), 'len: ', len(X_test[0]))
print('----start predict_proba-----')
preds = model.predict_proba(X_test)

of = open('./predictionaryion.csv', 'w')
csvWriter = csv.writer(of, delimiter=',')
csvWriter.writerow(['ID', 'Adoption', 'Died', 'Euthanasia', 'Return_to_owner' ,'Transfer'])
for i, row in enumerate(preds):
    csvWriter.writerow([i+1] + list(row))
of.close

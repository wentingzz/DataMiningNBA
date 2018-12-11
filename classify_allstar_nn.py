from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import numpy as np
import tensorflow as tf
import csv
from sklearn.preprocessing import MinMaxScaler


# variable declaration
np.random.seed(7)
tf.set_random_seed(7)
x_train = []
y_train = []
x_test = []
y_test = []
test_miss = []
x_label = []
scaler = MinMaxScaler()

# pre-processing for 2017-2018 data
with open('NBA Data - 2017-2018.csv', 'rb') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            x_label.append(row[2:24])
            x_label = x_label[0]
        else:
            line_count += 1
            x_test.append(row[2:24])
            if row[24] == "Yes":
                y_test.append([1])
            else:
                y_test.append([0])

# pre-processing for training data
with open('NBA Data - 2014-2015.csv', 'rb') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            line_count += 1
            x_train.append(row[2:24])
            if row[24] == "Yes":
                y_train.append([1])
            else:
                y_train.append([0])

with open('NBA Data - 2015-2016.csv', 'rb') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            line_count += 1
            x_train.append(row[2:24])
            if row[24] == "Yes":
                y_train.append([1])
            else:
                y_train.append([0])

with open('NBA Data - 2016-2017.csv', 'rb') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            line_count += 1
            x_train.append(row[2:24])
            if row[24] == "Yes":
                y_train.append([1])
            else:
                y_train.append([0])

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
dim = np.size(x_train, 1)
xt = np.asarray(x_train)
yt = np.asarray(y_train)
xts = np.asarray(x_test)
yts = np.asarray(y_test)

# create model
model = Sequential()
model.add(Dense(24, input_dim=dim, activation='elu'))
model.add(Dense(1, activation='relu'))

# compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# fit model
model.fit(xt, yt, epochs=10, batch_size=50, verbose=0)

test_scores = model.evaluate(xts, yts, batch_size=50, verbose=0)
print(test_scores[0] * 100)
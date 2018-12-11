#####################################################
#       TEAM NUMBER: 10                             #
#       TEAM MEMBER: Malik Majette  (mamajett)      #
#                    Qua Jones      (qyjones)       #
#                    Wenting Zheng  (wzheng8)       #
#####################################################
import numpy as np
import xlrd
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


#####################################################
#       This part is to read the player files       #
#####################################################

# Headers: 
xl = pd.ExcelFile("NBA Data.xlsx")
data14 = pd.read_excel(xl, sheet_name="2014-2015")
data15 = pd.read_excel(xl, sheet_name="2015-2016")
data16 = pd.read_excel(xl, sheet_name="2016-2017")
test17 = pd.read_excel(xl, sheet_name="2017-2018")

# Separate x and y for the training datasets
x_train14 = np.array(data14.loc[:, data14.columns[2:-1]])
x_train15 = np.array(data15.loc[:, data15.columns[2:-1]])
x_train16 = np.array(data16.loc[:, data16.columns[2:-1]])
# all three year data
x_train = np.append(x_train14, x_train15, axis = 0)
x_train = np.append(x_train, x_train16, axis = 0)

# x_train = np.array(data.loc[:,data.columns[2:-1]])

y_train14 = []
for data in data14['ALL-STAR']:
    if data == 'Yes':
        y_train14.append(1)
    else:
        y_train14.append(0)
y_train15 = []
for data in data15['ALL-STAR']:
    if data == 'Yes':
        y_train15.append(1)
    else:
        y_train15.append(0)
y_train16 = []
for data in data16['ALL-STAR']:
    if data == 'Yes':
        y_train16.append(1)
    else:
        y_train16.append(0)
y_train = np.append(y_train14, y_train15, axis = 0)
y_train = np.append(y_train, y_train16, axis = 0)


# Separate x and y for the testing dataset
x_test = np.array(test17.loc[:, test17.columns[2:-1]])
y_test = []
for data in test17['ALL-STAR']:
    if data == 'Yes':
        y_test.append(1)
    else:
        y_test.append(0)
# print(x_train16)
#####################################################
#           This part is for normalization          #
#####################################################
# normalize the training datasets
norm14 = (x_train14 - x_train14.min(axis=0)) / (x_train14.max(axis=0) - x_train14.min(axis=0))
norm15 = (x_train15 - x_train15.min(axis=0)) / (x_train15.max(axis=0) - x_train15.min(axis=0))
norm16 = (x_train16 - x_train16.min(axis=0)) / (x_train16.max(axis=0) - x_train16.min(axis=0))
norm_all = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0))

# normalize the testing dataset
normtest14 = (x_test - x_train14.min(axis=0)) / (x_train14.max(axis=0) - x_train14.min(axis=0))
normtest15 = (x_test - x_train15.min(axis=0)) / (x_train15.max(axis=0) - x_train15.min(axis=0))
normtest16 = (x_test - x_train16.min(axis=0)) / (x_train16.max(axis=0) - x_train16.min(axis=0))
normtest_all = (x_test - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0))

# from sklearn.preprocessing import StandardScaler  
# scaler = StandardScaler()  
# scaler.fit(x_train)
# 
# norm_all = scaler.transform(x_train)  
# normtest_all = scaler.transform(x_test)  
# print(norm_all)

# print(y_train)
#####################################################
#       This part is to fit the testing into        #
#       KNN model and calculate the error rate      #
#####################################################
def calerror(x_training, y_training, x_testing, y_testing, weight):
    k_list = np.arange(1,101, 1)
    error = []
    for k in k_list:
        if weight == 0:
            classifier = KNeighborsClassifier(p = 1, n_neighbors=k, weights = 'uniform')
        else:
            classifier = KNeighborsClassifier(p = 1, n_neighbors=k, weights = 'distance')
        classifier.fit(x_training, y_training)
        y_predicts = classifier.predict(x_testing)
        # for y_pre in y_predicts:
#             y_pre = 0 if y_pre <= 0.5 else 1
        error.append(float(sum(y_predicts != y_testing) / len(y_testing)))
    return error


#####################################################
#       This part is to plot the error tend         #
#####################################################
k_list = np.arange(1,101, 1)

err14 = calerror(norm14, y_train14, normtest14, y_test, 0)
err14_w = calerror(norm14, y_train14, normtest14, y_test, 1)
print(min(err14), ', ', min(err14_w))
print(max(err14), ', ', max(err14_w))

plt.plot(k_list, err14, 'k')
plt.plot(k_list, err14_w, 'k--')
plt.xlabel('k value')
plt.ylabel('error rate')
plt.show()

err15 = calerror(norm15, y_train15, normtest15, y_test, 0)
err15_w = calerror(norm15, y_train15, normtest15, y_test, 1)
print(min(err15), ', ', min(err15_w))
print(max(err15), ', ', max(err15))
plt.plot(k_list, err15, 'k')
plt.plot(k_list, err15_w, 'k--')
plt.xlabel('k value')
plt.ylabel('error rate')
plt.show()
# print(err16) 
err16 = calerror(norm16, y_train16, normtest16, y_test, 0)
err16_w = calerror(norm16, y_train16, normtest16, y_test, 1)
print(min(err16), ', ', min(err16_w))
print(max(err16), ', ', max(err16))
plt.plot(k_list, err16, 'k')
plt.plot(k_list, err16_w, 'k--')
plt.xlabel('k value')
plt.ylabel('error rate')
plt.show()
# print(err_all) 
err_all = calerror(norm_all, y_train, normtest_all, y_test, 0)
err_all_w = calerror(norm_all, y_train, normtest_all, y_test, 1)
print(min(err_all), ', ', min(err_all_w))
print(max(err_all), ', ', max(err_all))
plt.plot(k_list, err_all, 'k')
plt.plot(k_list, err_all_w, 'k--')
plt.xlabel('k value')
plt.ylabel('error rate')
plt.show()
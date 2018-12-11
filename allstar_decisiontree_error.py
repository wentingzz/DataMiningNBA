import pandas as pd
import numpy as np
import graphviz
from sklearn import tree
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

le = preprocessing.LabelEncoder()
plt.title('All Star Training Misclassification')
plt.xlabel('K Value')
plt.ylabel('Misclassification Rate')

df = pd.ExcelFile('NBA Data.xlsx')

data15 = df.parse('2014-2015')
data16 = df.parse('2015-2016')
data17 = df.parse('2016-2017')
data18 = df.parse('2017-2018')

x_train15 = np.array(data15.loc[:, data15.columns[2:-1]])
x_train16 = np.array(data16.loc[:, data16.columns[2:-1]])
x_train17 = np.array(data17.loc[:, data17.columns[2:-1]])

x_train = np.append(x_train15, x_train16, axis = 0)
x_train = np.append(x_train, x_train17, axis = 0)

y_train = []
for x in data15['ALL-STAR']:
    if x == 'Yes':
        y_train.append(1)
    else:
        y_train.append(0)

for x in data16['ALL-STAR']:
    if x == 'Yes':
        y_train.append(1)
    else:
        y_train.append(0)

for x in data17['ALL-STAR']:
    if x == 'Yes':
        y_train.append(1)
        class_names.append('Yes')
    else:
        y_train.append(0)
        class_names.append('No')


x_test = np.array(data18.loc[:, data18.columns[2:-1]])
y_test = []
for x in data18['ALL-STAR']:
    if x == 'Yes':
        y_test.append(1)
    else:
        y_test.append(0)

clf_gini = tree.DecisionTreeClassifier(criterion='gini')
clf_entr = tree.DecisionTreeClassifier(criterion='entropy')
plt.style.use('ggplot')

for x in range(2,11):
    gini_scores = cross_val_score(clf_gini, x_train, y_train, cv=x)
    entr_scores = cross_val_score(clf_entr, x_train, y_train, cv=x)
    for i in np.nditer(gini_scores, op_flags=['readwrite']):
        i[...] = 1 - i
    for i in np.nditer(entr_scores, op_flags=['readwrite']):
        i[...] = 1 - i
    plt.plot(x,(np.mean(gini_scores)),'go')
    plt.plot(x,(np.mean(entr_scores)),'bo')
plt.legend(("Gini Impurity", "Entropy"))
plt.show()
plt.close()

clf_gini.fit(x_train, y_train)
clf_entr.fit(x_train, y_train)

print("GINI: ")
y_pred = clf_gini.predict(np.array(x_test))
print("Error Rate: %f" % ((1 - accuracy_score(y_test, y_pred))))
print("Report: ")
print(classification_report(y_test, y_pred))
#
print("Entropy: ")
y_pred = clf_entr.predict(np.array(x_test))
print("Error Rate: %f" % ((1 - accuracy_score(y_test, y_pred))))
print("Report: ")
print(classification_report(y_test, y_pred))

cols = ['AGE','GP', 'W', 'L', 'MIN', 'PTS',	'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%','OREB','DREB','REB','AST','TOV','STL','BLK']
dot_data = tree.export_graphviz(clf_gini, out_file=None, filled=True, feature_names=cols, class_names=None)
graph = graphviz.Source(dot_data)
graph.render('All Star Classifier: Gini')
dot_data = tree.export_graphviz(clf_entr, out_file=None, filled=True, feature_names=cols, class_names=None)
graph = graphviz.Source(dot_data)
graph.render('All Star Classifier: Entropy')

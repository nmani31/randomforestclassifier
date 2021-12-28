import pandas as pd
dataset = pd.read_csv('dataset-pathname.csv')
from sklearn.model_selection import train_test_split
independent = dataset[['VOP','NOP','POP']]
dependent = dataset['motility']
dependent_train, dependent_test, independent_train, independent_test = train_test_split(dependent, independent, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200)
clf.fit(independent_train,dependent_train)
from sklearn import metrics
a = []
b = []
for i in range (10000):
    dependent_pred = clf.predict(independent_test)
    accuracy = metrics.accuracy_score(dependent_pred, dependent_test)
    a.append(accuracy)
    precision = metrics.precision_score(dependent_pred, dependent_test)
    b.append(precision)
    i = i + 1
print('Accuracy', sum(a)/len(a))
print('Precision', sum(b)/len(b))

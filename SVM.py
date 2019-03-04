from sklearn import svm, preprocessing
import pandas as pd
from sklearn.metrics import accuracy_score

global last_test_accuracy

def split_data():
    df = pd.read_csv('data.csv')
    le = preprocessing.LabelEncoder()
    for col_name in df.columns:
        if df[col_name].dtype == object:
            df[col_name] = le.fit_transform(df[col_name])
        else:
            pass
    Y = df['total-per-year']
    X = df.drop('total-per-year', 1)

    TRAIN_SIZE = int(round(X.size*0.7))
    x_train = X[:TRAIN_SIZE]
    y_train = Y[:TRAIN_SIZE]
    x_test = X[TRAIN_SIZE:]
    y_test = Y[TRAIN_SIZE:]
    return x_train, x_test, y_train, y_test

def svm_train_test(x_train, x_test, y_train, y_test):
    clf = svm.SVC(kernel='linear',gamma=1,C=1000)
    clf.fit(x_train, y_train)
    PREDICTED = clf.predict(x_test)

    global last_test_accuracy
    last_test_accuracy = accuracy_score(y_test, PREDICTED)
    return last_test_accuracy
from sklearn import svm, preprocessing
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.metrics import accuracy_score


class Training_Object():
    #Constructor
    def __init__(self):
        self.sample_count = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.svm_accuracy = None
        self.nb_accuracy = None
    #Method that obtain the data from data.csv and split it in then way that is needed for trainning the model
    def split_data(self):
        df = pd.read_csv('data.csv')
        le = preprocessing.LabelEncoder()
        for col_name in df.columns:
            if df[col_name].dtype == object:
                df[col_name] = le.fit_transform(df[col_name])
            else:
                pass
        Y = df['total-per-year']
        X = df.drop('total-per-year', 1)

        TRAIN_SIZE = int(round(X.shape[0]*0.7))
        self.sample_count = Y.size
        self.x_train = X[:TRAIN_SIZE]
        self.y_train = Y[:TRAIN_SIZE]
        self.x_test = X[TRAIN_SIZE:]
        self.y_test = Y[TRAIN_SIZE:]
    
    #Train and test the svm model
    def svm_train_test(self):
        clf = svm.SVC(kernel='linear',gamma=1,C=1)
        clf.fit(self.x_train, self.y_train)
        PREDICTED = clf.predict(self.x_test)

        self.svm_accuracy = accuracy_score(self.y_test, PREDICTED)
    
    def nb_train_test(self):
        clf = GaussianNB()
        clf.fit(self.x_train, self.y_train)
        PREDICTED = clf.predict(self.x_test)

        self.nb_accuracy = accuracy_score(self.y_test, PREDICTED)

    def train_test(self):
        self.svm_train_test()
        self.nb_train_test()
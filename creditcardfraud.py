import pandas as pd
from sklearn.preprocessing import MinMaxScaler

datasets = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
labels = datasets.loc[:,'Class']
datasets.drop(['Class'], axis=1, inplace=True)
from dbn import SupervisedDBNClassification, SupervisedDBNRegression
X_train, X_test, y_train, y_test = train_test_split(datasets,labels, test_size = 0.20, random_state = 42)
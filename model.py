import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from preprocessing import preprocess


df_train = pd.read_csv('data/train.csv')
X,y = preprocess(df_train)
def model():
    # I apply the random
    clf = RandomForestClassifier()

    # I split the data to train it
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf.fit(X_train, y_train)


    #I ll save my model
    joblib.dump(clf,'model.pkl')
    print("Model dumped !")

    # Load the model that you just saved
    clf = joblib.load('model.pkl')

    model_columns = list(X.columns)
    joblib.dump(model_columns, 'model_columns.pkl')
    print("Models columns dumped!")

    return clf.score(X_test, y_test)
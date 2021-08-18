import pandas as pd


def preprocess(df_train):
    #Here i'll try to fill NaN values of age by is Mode
    df_train.Age.mode()
    df_train.Age = df_train.Age.fillna(float(df_train.Age.mode()))

    df_train = df_train.drop(columns=['Cabin'])

    nan_value = float("NaN")
    df_train.replace("", nan_value, inplace=True)

    df_train.dropna(subset=['Embarked'],inplace=True)

    #Make Dummies
    gender_dummies = pd.get_dummies(df_train.Sex,drop_first=True)
    embarked_dummies = pd.get_dummies(df_train.Embarked)

    #Drop the Ones I already Dummies
    df_train = df_train.drop(columns=['Sex','Embarked'])
    df_train = pd.concat([df_train,gender_dummies,embarked_dummies],axis=1)

    # I do remove cabin 'cause I don't see the importance of this column..
    df_trained = df_train.drop(columns=['Ticket','Fare','Name','PassengerId','Pclass'])

    X = df_trained.drop('Survived',axis=1)
    y = df_trained['Survived']

    return X,y;
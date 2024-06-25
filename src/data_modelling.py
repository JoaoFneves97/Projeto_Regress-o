import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class Model_Regression:
    def model_sep(data_norm):
        X = data_norm.drop(columns = ["loan_status"])
        y = data_norm["loan_status"]
        
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 42)
        return X_train, X_test, y_train, y_test

    def create_model(X_train, X_test, y_train, y_test):
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        return y_pred
    
    def model_data(data_norm):
        X_train, X_test, y_train, y_test = Model_Regression.model_sep(data_norm)
        y_pred = Model_Regression.create_model(X_train, X_test, y_train, y_test)
        return X_train,X_test, y_train, y_test, y_pred


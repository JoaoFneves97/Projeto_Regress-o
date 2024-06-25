from src.data_preparation import Preparation
from src.data_collection import load_data
from src.data_preprocessing import Preprocessing
from src.data_modelling import Model_Regression
from src.model_evaluation import Evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import pandas as pd
import numpy as np


def save_data(dataframe, file_path):
    dataframe.to_csv(file_path, index=False)

def main(caminho_arquivo):

    data = load_data(caminho_arquivo)

    data = Preparation.prepare_data(data)

    data_norm = Preprocessing.preprocessing_data(data)
    
    save_data(data_norm, "data/data_preparado.csv")

    X_train,X_test, y_train, y_test, y_pred = Model_Regression.model_data(data_norm)

    evaluation_of_model = Evaluation(y_test, y_pred)

    evaluation_of_model


# TUNAGEM DO MODELO
    pipeline = Pipeline([
    ('logreg', LogisticRegression(max_iter=1000))
    ])

    param_grid = {
    'logreg__penalty': ['l1', 'l2'],
    'logreg__C': np.logspace(-4, 4, 20),
    'logreg__solver': ['liblinear', 'saga']
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f'Melhores hiperparâmetros: {best_params}')

# APLICAÇÃO DA TUNAGEM DOS MELHORES PARAMETROS
    final_model = LogisticRegression(
        penalty=best_params['logreg__penalty'],
        C=best_params['logreg__C'],
        solver=best_params['logreg__solver'],
        max_iter=1000
    )

    final_pipeline = Pipeline([
        ('logreg', final_model)
    ])

    final_pipeline.fit(X_train, y_train)


    y_pred_tunned = final_pipeline.predict(X_test)

# AVALIAÇÃO DO MODELO TUNADO
    accuracy = accuracy_score(y_test, y_pred_tunned)
    print(f'A Acuracia do modelo foi de: {(accuracy*100):.2f}%')
    print(" ")

    y_proba = final_pipeline.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test,y_proba)
    print(f"Roc Auc do modelo foi de: {(roc*100):.2f}%")

if __name__ == "__main__":
    caminho_arquivo = "data/loan_approval_dataset.csv"
    main(caminho_arquivo)
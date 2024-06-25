from src.data_preparation import Preparation
from src.data_collection import load_data
from src.data_preprocessing import Preprocessing
from src.data_modelling import Model_Regression
from src.model_evaluation import Evaluation

def main(caminho_arquivo):

    data = load_data(caminho_arquivo)

    data = Preparation.prepare_data(data)

    data_norm = Preprocessing.preprocessing_data(data)

    y_test, y_pred = Model_Regression.model_data(data_norm)

    Evaluation(y_test, y_pred)


if __name__ == "__main__":
    caminho_arquivo = "data/loan_approval_dataset.csv"
    main(caminho_arquivo)
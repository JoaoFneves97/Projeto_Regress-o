import pandas as pd

class Preparation:
    def empty_names(data):
        data.columns = data.columns.str.strip()
        for col in data.select_dtypes(['object']).columns:
            data[col] = data[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        data = data.drop(["loan_id"], axis=1)
        return data

    def balanced_target(data):
        classe_majo = data[data["loan_status"] == "Approved"]
        classe_mino = data[data["loan_status"] == "Rejected"]

        subamostragem = classe_majo.sample(n = len(classe_mino))

        data = pd.concat([subamostragem,classe_mino])

        data = data.sample(frac=1).reset_index(drop=True)
        return data
    
    def prepare_data(data):
        data = Preparation.empty_names(data)
        data = Preparation.balanced_target(data)
        return data
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler

class Preprocessing:
    def encode_categorical_columns(data):
        le = LabelEncoder()
        for col in data.select_dtypes(['object']).columns:
            data[col] = le.fit_transform(data[col])
        return data

    def normalize(data):
        scaler = StandardScaler()
        df_norm = scaler.fit_transform(data)
        return pd.DataFrame(df_norm, columns = data.columns)
    
    def preprocessing_data(data):
        data = Preprocessing.encode_categorical_columns(data)
        data_norm = Preprocessing.normalize(data)
        return data_norm




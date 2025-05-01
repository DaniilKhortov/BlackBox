import pandas as pd
import io
import string
import os
from typing import Dict, Optional, Union, List
from collections import Counter
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

def load_data_from_mongo(
    db_name: str,
    collection_name: str,
    required_columns: List[str],
    connection_string: str = "mongodb://localhost:27017/"
) -> Optional[pd.DataFrame]:

    print(f"Спроба завантажити дані з MongoDB: {db_name}.{collection_name}...")
    dataframe = None
    client = None 

    try:
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        client.server_info() 
        db = client[db_name]
        collection = db[collection_name]

        projection = {col: 1 for col in required_columns}
        projection['_id'] = 0

        cursor = collection.find({}, projection)
        data_list = list(cursor)

        if not data_list:
            print(f"Попередження: В колекції '{collection_name}' не знайдено даних з полями {required_columns}.")
            return None

        dataframe = pd.DataFrame(data_list)
        print(f"Прочитано {len(dataframe)} записів.")

    except ConnectionFailure:
        print(f"Помилка: Не вдалося підключитися до MongoDB '{connection_string}'.")
        return None
    except Exception as e:
        print(f"Помилка під час роботи з MongoDB: {e}")
        return None
    finally:
        if client: 
            client.close()

    if dataframe is None or dataframe.empty: 
        print("Попередження: Не вдалося завантажити дані або дані порожні.")
        return None

    if not all(col in dataframe.columns for col in required_columns):
        missing = [col for col in required_columns if col not in dataframe.columns]
        print(f"Помилка: У завантажених даних відсутні колонки: {missing}.")
        return None

    df_selected = dataframe[required_columns].copy() 
    df_selected.dropna(inplace=True)

    if 'Original' in df_selected.columns:
        df_selected['Original'] = df_selected['Original'].astype(str)
    if 'Encripted' in df_selected.columns:
        df_selected['Encripted'] = df_selected['Encripted'].astype(str)
    if 'Slide' in df_selected.columns:
         try:
             df_selected['Slide'] = pd.to_numeric(df_selected['Slide'], errors='coerce').astype('Int64')
             df_selected.dropna(subset=['Slide'], inplace=True)
         except Exception as e:
              print(f"Помилка конвертації 'Slide' в число: {e}")
              return None

    if df_selected.empty:
         print("Попередження: Після очищення не залишилось валідних даних.")
         return None

    print(f"Успішно завантажено та оброблено {len(df_selected)} записів.")
    return df_selected


class AtbashDecryptor: 
    def __init__(self, db_name: str, collection_name: str, connection_string: str = "mongodb://localhost:27017/"):
        self.learned_pairs: Dict[str, List[str]] = {char: [] for char in string.ascii_lowercase}
        required_cols = ['Original', 'Encripted']

        dataframe = load_data_from_mongo(db_name, collection_name, required_cols, connection_string)

        if dataframe is not None:
            self._train_model(dataframe) 
        else:
             print(f"Попередження: Не вдалося завантажити дані для {db_name}.{collection_name}. Модель Атбаш буде порожньою.")

    def _train_model(self, df: pd.DataFrame):
        print("Навчання моделі: збір пар символів...")
        count = 0
        for _, row in df.iterrows():
            original = row['Original'].lower(); encrypted = row['Encripted'].lower()
            if len(original) != len(encrypted): continue
            for i in range(len(original)):
                orig_char = original[i]; enc_char = encrypted[i]
                if 'a' <= enc_char <= 'z' and 'a' <= orig_char <= 'z':
                    self.learned_pairs[enc_char].append(orig_char); count += 1
        print(f"Навчання завершено. Зібрано {count} пар символів з даних.")

    def _predict_char(self, enc_char_lower: str) -> Optional[str]:
        if enc_char_lower in self.learned_pairs and self.learned_pairs[enc_char_lower]:
            most_common_pair = Counter(self.learned_pairs[enc_char_lower]).most_common(1)
            if most_common_pair:
                return most_common_pair[0][0]
        return None 

    def decrypt(self, encrypted_text: str) -> str:
        decrypted_text = ""
        for char in encrypted_text:
            predicted_char = None 
            if 'a' <= char.lower() <= 'z':
                lower_char = char.lower()

                predicted_char = self._predict_char(lower_char)

            if predicted_char:
                decrypted_text += predicted_char.upper() if char.isupper() else predicted_char
            else:
                decrypted_text += char
        return decrypted_text




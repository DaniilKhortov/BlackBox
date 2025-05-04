import pandas as pd
import io
import string
import os
from typing import Dict, Optional, Union, List, Set, Tuple
from collections import Counter # Потрібен для частотного аналізу
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

LOWERCASE_LETTERS = string.ascii_lowercase
ENGLISH_FREQUENCIES = { 
    'a': 0.0817, 'b': 0.0150, 'c': 0.0278, 'd': 0.0425, 'e': 0.1270, 'f': 0.0223,
    'g': 0.0202, 'h': 0.0609, 'i': 0.0697, 'j': 0.0015, 'k': 0.0077, 'l': 0.0403,
    'm': 0.0241, 'n': 0.0675, 'o': 0.0751, 'p': 0.0193, 'q': 0.0010, 'r': 0.0599,
    's': 0.0633, 't': 0.0906, 'u': 0.0276, 'v': 0.0098, 'w': 0.0236, 'x': 0.0015,
    'y': 0.0197, 'z': 0.0007
}

def calculate_frequencies(text: str) -> Dict[str, float]:
    text_lower = text.lower()
    letter_counts = Counter(c for c in text_lower if c in LOWERCASE_LETTERS)
    total_letters = sum(letter_counts.values())
    if total_letters == 0:
        return {char: 0.0 for char in LOWERCASE_LETTERS}
    frequencies = {char: letter_counts.get(char, 0) / total_letters for char in LOWERCASE_LETTERS}
    return frequencies

def score_frequency_match(text_freq: Dict[str, float]) -> float:
    score = 0.0
    for char in LOWERCASE_LETTERS:
        score += text_freq.get(char, 0.0) * ENGLISH_FREQUENCIES.get(char, 0.0)
    return score

# --- Клас Дешифратора Цезаря---
class CaesarDecryptor:
    EXPECTED_SHIFTS = set(range(1, 26))

    def __init__(self, db_name: str, collection_name: str, connection_string: str = "mongodb://localhost:27017/"):
        self.learned_mappings: Dict[int, Dict[str, str]] = {}
        required_cols = ['Original', 'Slide', 'Encripted']
        self.EXPECTED_SHIFTS = set(range(1, 26))

        dataframe = load_data_from_mongo(db_name, collection_name, required_cols, connection_string)

        if dataframe is not None:
            self._train_model(dataframe) 
        else:
             print(f"Попередження: Не вдалося завантажити дані для {db_name}.{collection_name}. Модель Цезаря буде порожньою.")

    def _train_model(self, df: pd.DataFrame):
        print("Навчання моделі Цезаря: збір відповідностей...")
        for _, row in df.iterrows():
            original_word = str(row['Original']).lower(); encrypted_word = str(row['Encripted']).lower()
            shift = row['Slide']
            if pd.isna(shift) or shift not in self.EXPECTED_SHIFTS: continue
            if len(original_word) != len(encrypted_word): continue
            current_mapping = self.learned_mappings.setdefault(shift, {})
            for i in range(len(original_word)):
                orig_char, enc_char = original_word[i], encrypted_word[i]
                if 'a' <= enc_char <= 'z' and 'a' <= orig_char <= 'z':
                    if enc_char in current_mapping and current_mapping[enc_char] != orig_char: pass # Попередження можна прибрати
                    current_mapping[enc_char] = orig_char
        print("Навчання завершено.")
        if self.learned_mappings: print(f"Вивчено відповідності для зсувів: {sorted(list(self.learned_mappings.keys()))}")
        else: print("Модель порожня.")

    def _try_decrypt_with_map(self, encrypted_text: str, shift_map: Dict[str, str]) -> str:
        decrypted_text = ""
        for char in encrypted_text:
            decrypted_char = char 
            if 'a' <= char.lower() <= 'z':
                lower_char = char.lower()
                predicted_original_char = shift_map.get(lower_char)
                if predicted_original_char:
                    decrypted_char = predicted_original_char.upper() if char.isupper() else predicted_original_char
            decrypted_text += decrypted_char
        return decrypted_text

    def decrypt(self, encrypted_text: str) -> Tuple[Optional[int], str]:
        if not self.learned_mappings:
            print("Помилка: Модель не навчена (немає даних). Автовизначення неможливе.")
            return None, encrypted_text

        best_shift = None
        best_score = -1.0 
        best_decryption_attempt = encrypted_text 

        for shift, current_mapping in self.learned_mappings.items():
             decryption_attempt = self._try_decrypt_with_map(encrypted_text, current_mapping)

             attempt_frequencies = calculate_frequencies(decryption_attempt)
             current_score = score_frequency_match(attempt_frequencies)

             # print(f"DEBUG: Зсув {shift}, Оцінка {current_score:.4f}, Спроба: '{decryption_attempt[:50]}...'") # Для відладки

             if current_score > best_score:
                 best_score = current_score
                 best_shift = shift
                 best_decryption_attempt = decryption_attempt

        MIN_ACCEPTABLE_SCORE = 0.03 

        if best_score < MIN_ACCEPTABLE_SCORE:
            print(f"Попередження: Найкраща оцінка ({best_score:.4f}) занадто низька. "
                  f"Не вдалося надійно визначити зсув.")
            return None, encrypted_text 
        else:
             print(f"Визначено зсув: {best_shift} (Оцінка: {best_score:.4f})")
             return best_shift, best_decryption_attempt

import pandas as pd
import io
import string
import os
import re
from typing import Dict, Optional, Union, List, Set
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import nltk
try:
    from nltk.corpus import words
    ENGLISH_WORDS: Set[str] = set(w.lower() for w in words.words())
    print(f"NLTK: Завантажено {len(ENGLISH_WORDS)} слів (у нижньому регістрі).")
except ImportError:
    print("Помилка: Бібліотека NLTK не встановлена. Запустіть 'pip install nltk'.")
    ENGLISH_WORDS = set()
except LookupError:
    print("Помилка: Корпус NLTK 'words' не завантажено.")
    print("Запустіть Python і виконайте: import nltk; nltk.download('words')")
    ENGLISH_WORDS = set()

def load_data_from_mongo(
    db_name: str,
    collection_name: str,
    required_columns: List[str],
    connection_string: str = "mongodb://localhost:27017/"
) -> Optional[pd.DataFrame]:

    #print(f"Спроба завантажити дані з MongoDB: {db_name}.{collection_name}...")
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


# --- Клас Дешифратора Свинячої Латини  ---
class PigLatinDecryptor: 

    VOWELS = "aeiou"

    def __init__(self, db_name: str, collection_name: str, connection_string: str = "mongodb://localhost:27017/"):
        """
        Ініціалізує дешифратор, завантажуючи дані ТІЛЬКИ з MongoDB.
        """
        self.learned_word_map: Dict[str, str] = {}
        required_cols = ['Original', 'Encripted']
        self.VOWELS = "aeiou" 

        dataframe = load_data_from_mongo(db_name, collection_name, required_cols, connection_string)

        if dataframe is not None:
            self._train_model(dataframe) 
        else:
             print(f"Попередження: Не вдалося завантажити дані для {db_name}.{collection_name}. Модель Свинячої Латини буде порожньою (тільки словник NLTK).")

    def _train_model(self, df: pd.DataFrame):
        print("Навчання моделі: збір пар слів...")
        count = 0; duplicates_found = 0
        for _, row in df.iterrows():
            original_word = row['Original']; encrypted_word = row['Encripted']
            if not original_word or not encrypted_word: continue
            encrypted_key = encrypted_word.lower()
            if encrypted_key in self.learned_word_map:
                if self.learned_word_map[encrypted_key] != original_word: duplicates_found +=1
                self.learned_word_map[encrypted_key] = original_word
            else:
                self.learned_word_map[encrypted_key] = original_word; count += 1
        print(f"Навчання завершено. Вивчено {count} унікальних пар слів.")
        if duplicates_found > 0: print(f"Знайдено та перезаписано {duplicates_found} дублікатів.")

    def _is_vowel(self, char: str) -> bool:
        return char in self.VOWELS

    def _generate_decryption_candidates(self, encrypted_word: str) -> List[str]:
        """
        Генерує список можливих оригінальних слів у нижньому регістрі.
        """
        lower_word = encrypted_word.lower()
        candidates = []

        # Правило '-yay'
        if len(lower_word) > 3 and lower_word.endswith('yay'):
            candidates.append(lower_word[:-3])
            return candidates

        # Правило '-ay'
        elif len(lower_word) > 2 and lower_word.endswith('ay'):
            stem = lower_word[:-2]
            if not stem: return [] # Порожній stem - немає кандидатів

            # Якщо stem закінчується на голосну, кандидат - це сам stem
            if self._is_vowel(stem[-1]):
                 candidates.append(stem)

            max_possible_moved = 0
            for i in range(len(stem) -1, -1, -1):
                 if not self._is_vowel(stem[i]):
                     max_possible_moved += 1
                 else:
                     break

            for k in range(1, max_possible_moved + 1):
                if k <= len(stem):
                    moved_consonants = stem[-k:]
                    base = stem[:-k]
                    candidates.append(moved_consonants + base)
            return list(set(candidates)) 

        return []

    def decrypt(self, encrypted_text: str) -> str:
        parts = re.split(r'(\b\w+\b)', encrypted_text)
        decrypted_parts = []

        for part in parts:
            decrypted_word = part 
            is_processed = False

            if part and re.match(r'\b\w+\b', part): 
                word_to_process = part
                lower_word_key = word_to_process.lower()

                if lower_word_key in self.learned_word_map:
                    original_word_from_map = self.learned_word_map[lower_word_key]
                    if word_to_process[0].isupper() and original_word_from_map:
                         decrypted_word = original_word_from_map[0].upper() + original_word_from_map[1:].lower()
                    elif word_to_process.isupper() and original_word_from_map:
                         decrypted_word = original_word_from_map.upper()
                    else: decrypted_word = original_word_from_map
                    is_processed = True
                if not is_processed and ENGLISH_WORDS:
                    candidates_lower = self._generate_decryption_candidates(word_to_process)

                    for guess_lower in candidates_lower:
                        if guess_lower in ENGLISH_WORDS:
                            if word_to_process[0].isupper():
                                decrypted_word = guess_lower[0].upper() + guess_lower[1:]
                            elif word_to_process.isupper():
                                 decrypted_word = guess_lower.upper()
                            else:
                                 decrypted_word = guess_lower
                            is_processed = True
                            break 

            decrypted_parts.append(decrypted_word)

        return "".join(decrypted_parts)




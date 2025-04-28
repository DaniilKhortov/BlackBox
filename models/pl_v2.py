import pandas as pd
import io
import string
import os
import re
from typing import Dict, Optional, Union, List, Set

import nltk
try:
    from nltk.corpus import words
    # Важливо: сет з ЛИШЕ МАЛЕНЬКИМИ літерами 
    ENGLISH_WORDS: Set[str] = set(w.lower() for w in words.words())
    print(f"NLTK: Завантажено {len(ENGLISH_WORDS)} слів (у нижньому регістрі).")
except ImportError:
    print("Помилка: Бібліотека NLTK не встановлена. Запустіть 'pip install nltk'.")
    ENGLISH_WORDS = set()
except LookupError:
    print("Помилка: Корпус NLTK 'words' не завантажено.")
    print("Запустіть Python і виконайте: import nltk; nltk.download('words')")
    ENGLISH_WORDS = set()

def load_words_from_csv(file_path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(file_path)
        if df.empty: return None
        required_columns = ['Original', 'Encripted']
        if not all(col in df.columns for col in required_columns):
            print(f"Помилка: У файлі '{file_path}' відсутні необх. колонки ({', '.join(required_columns)}).")
            return None
        df_selected = df[required_columns].copy(); df_selected.dropna(inplace=True)
        df_selected['Original'] = df_selected['Original'].astype(str); df_selected['Encripted'] = df_selected['Encripted'].astype(str)
        if df_selected.empty: return None
        return df_selected
    except FileNotFoundError: print(f"Помилка: Файл '{file_path}' не знайдено."); return None
    except pd.errors.EmptyDataError: print(f"Помилка: Файл '{file_path}' порожній."); return None
    except Exception as e: print(f"Помилка читання '{file_path}': {e}"); return None


# --- Клас Дешифратора Свинячої Латини ---
class PigLatinSmartCandidateDecryptor: 
    VOWELS = "aeiou"

    def __init__(self, source: Union[str, pd.DataFrame], is_file_path: bool = False):
        self.learned_word_map: Dict[str, str] = {}
        dataframe = self._load_dataframe(source, is_file_path)
        if dataframe is not None:
            self._train_model(dataframe)
        else:
             print("Попередження: Немає даних CSV. Покладання лише на алгоритм+словник.")

    def _load_dataframe(self, source: Union[str, pd.DataFrame], is_file_path: bool) -> Optional[pd.DataFrame]:
        dataframe = None
        if is_file_path:
            if isinstance(source, str): dataframe = load_words_from_csv(source)
            else: print("Помилка ініціалізації: Якщо is_file_path=True, 'source' має бути рядком.")
        else:
            if isinstance(source, str):
                try:
                    dataframe = pd.read_csv(io.StringIO(source))
                    required_columns = ['Original', 'Encripted'];
                    if not all(col in dataframe.columns for col in required_columns): return None
                    if not dataframe.empty:
                        dataframe = dataframe[required_columns].copy(); dataframe.dropna(inplace=True)
                        dataframe['Original'] = dataframe['Original'].astype(str); dataframe['Encripted'] = dataframe['Encripted'].astype(str)
                        if dataframe.empty: return None
                    else: return None
                except Exception: return None
            elif isinstance(source, pd.DataFrame):
                 required_columns = ['Original', 'Encripted']
                 if all(col in source.columns for col in required_columns):
                     dataframe = source[required_columns].copy(); dataframe.dropna(inplace=True)
                     dataframe['Original'] = dataframe['Original'].astype(str); dataframe['Encripted'] = dataframe['Encripted'].astype(str)
                     if dataframe.empty: return None
                 else: return None
            else: print("Помилка ініціалізації: Непідтримуваний тип 'source'."); return None
        return dataframe

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
        lower_word = encrypted_word.lower()
        candidates = []

        # Правило '-yay'
        if len(lower_word) > 3 and lower_word.endswith('yay'):
            candidates.append(lower_word[:-3])
            return candidates 

        # Правило '-ay'
        elif len(lower_word) > 2 and lower_word.endswith('ay'):
            stem = lower_word[:-2]
            if not stem: return [] 

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

            return list(set(candidates)) # Повертаємо унікальних кандидатів

        return []

    def decrypt(self, encrypted_text: str) -> str:
        """
        Дешифрує текст, використовуючи мапу CSV або перебираючи
        кандидатів від алгоритму та перевіряючи їх словником NLTK.
        """
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
                            break # Зупиняємось на першому вдалому кандидаті


            decrypted_parts.append(decrypted_word)

        return "".join(decrypted_parts)


# --- Приклад використання  ---
if __name__ == "__main__":
    temp_file_path = "D:/Mysor2/3kurs/Coll_proc_data/Project/NewGen/EnigmaticCodes.PL.csv"

    print("\n--- Тест 1: Ініціалізація з ШЛЯХУ до файлу (Candidate клас) ---")
    pl_decryptor = PigLatinSmartCandidateDecryptor(temp_file_path, is_file_path=True)

    print("\nДешифрування (CSV + Кандидати + Словник):")
    test_pl_text1 = "oetay xtaujay eppgrziidkyay."
    print(f"'{test_pl_text1}' -> '{pl_decryptor.decrypt(test_pl_text1)}'")

    test_pl_text2 = "Appleyay ingstray!"
    print(f"'{test_pl_text2}' -> '{pl_decryptor.decrypt(test_pl_text2)}'") # Очікуємо: 'Apple string!'

    test_pl_text3 = "Ythonpay isyay oolcay."
    print(f"'{test_pl_text3}' -> '{pl_decryptor.decrypt(test_pl_text3)}'") # Очікуємо: 'Python is cool.'

    test_pl_text4 = "Kay oetay Zay. Iyay."
    print(f"'{test_pl_text4}' -> '{pl_decryptor.decrypt(test_pl_text4)}'")

    test_pl_text5 = "esttay ordway"
    print(f"'{test_pl_text5}' -> '{pl_decryptor.decrypt(test_pl_text5)}'") # Очікуємо: 'test word'

    test_pl_text6 = "ivegay emay ayay hurchcay"
    print(f"'{test_pl_text6}' -> '{pl_decryptor.decrypt(test_pl_text6)}'") # Очікуємо: 'give me a church'

    test_pl_text7 = "Igpay Atinlay isyay unfay!"
    print(f"'{test_pl_text7}' -> '{pl_decryptor.decrypt(test_pl_text7)}'") # Очікуємо: 'Pig Latin is fun!'

    # Кандидат 'cabxyz' не буде в словнику
    test_pl_text8 = "xyzabcay"
    print(f"'{test_pl_text8}' -> '{pl_decryptor.decrypt(test_pl_text8)}'") # Очікуємо: 'xyzabcay'

    print("\n--- Тест з некоректним шляхом (Candidate клас) ---")
    pl_decryptor_bad = PigLatinSmartCandidateDecryptor("non_existent_file.csv", is_file_path=True)
    print("\nДешифрування (тільки Кандидати + Словник):")
    print(f"'{test_pl_text5}' -> '{pl_decryptor_bad.decrypt(test_pl_text5)}'") # Очікуємо: 'test word'
    print(f"'{test_pl_text7}' -> '{pl_decryptor_bad.decrypt(test_pl_text7)}'") # Очікуємо: 'Pig Latin is fun!'
    print(f"'{test_pl_text8}' -> '{pl_decryptor_bad.decrypt(test_pl_text8)}'") # Очікуємо: 'xyzabcay'
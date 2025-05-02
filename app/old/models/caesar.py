import pandas as pd
import io
import string
import os
from typing import Dict, Optional, Union, List, Set

# --- Функція завантаження CSV  ---
def load_caesar_data_from_csv(file_path: str) -> Optional[pd.DataFrame]:
    """
    Читає CSV-файл для Цезаря та повертає дані з колонок
    'Original', 'Slide', та 'Encripted'.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty: return None
        required_columns = ['Original', 'Slide', 'Encripted']
        if not all(col in df.columns for col in required_columns):
            print(f"Помилка: У файлі '{file_path}' відсутні необх. колонки ({', '.join(required_columns)}).")
            return None
        df_selected = df[required_columns].copy()
        df_selected.dropna(inplace=True)
        try:
            df_selected['Slide'] = pd.to_numeric(df_selected['Slide'], errors='coerce').astype('Int64')
            df_selected.dropna(subset=['Slide'], inplace=True) 
        except Exception as e:
             print(f"Помилка конвертації колонки 'Slide' в число: {e}")
             return None

        df_selected['Original'] = df_selected['Original'].astype(str)
        df_selected['Encripted'] = df_selected['Encripted'].astype(str)
        if df_selected.empty: return None
        print(f"Успішно завантажено {len(df_selected)} записів з файлу '{file_path}'.")
        return df_selected
    except FileNotFoundError: print(f"Помилка: Файл '{file_path}' не знайдено."); return None
    except pd.errors.EmptyDataError: print(f"Помилка: Файл '{file_path}' порожній."); return None
    except Exception as e: print(f"Помилка читання '{file_path}': {e}"); return None

# --- Клас Дешифратора Цезаря  ---
class CaesarLearnedDecryptor:

    EXPECTED_SHIFTS = {1, 2, 3, 4} # Очікувані значення зсуву

    def __init__(self, source: Union[str, pd.DataFrame], is_file_path: bool = False):
        """
        Ініціалізує дешифратор та "навчає" його, створюючи
        окремі мапи символів для кожного зсуву.
        """
        # Структура: { shift: {encrypted_char: original_char, ...}, ... }
        self.learned_mappings: Dict[int, Dict[str, str]] = {}
        dataframe = self._load_dataframe(source, is_file_path)

        if dataframe is not None:
            self._train_model(dataframe)
        else:
             print("Попередження: Не вдалося завантажити дані для навчання. Модель буде порожньою.")

    def _load_dataframe(self, source: Union[str, pd.DataFrame], is_file_path: bool) -> Optional[pd.DataFrame]:
        """Завантажує DataFrame з джерела, використовуючи адаптовану функцію."""
        dataframe = None
        if is_file_path:
            if isinstance(source, str):
                dataframe = load_caesar_data_from_csv(source)
            else:
                 print("Помилка ініціалізації: Якщо is_file_path=True, 'source' має бути рядком.")
        else:
            if isinstance(source, str):
                try:
                    dataframe = pd.read_csv(io.StringIO(source))
                    required_columns = ['Original', 'Slide', 'Encripted']
                    if not all(col in dataframe.columns for col in required_columns):
                        print(f"Помилка: У рядку CSV відсутні колонки ({', '.join(required_columns)}).")
                        return None
                    if not dataframe.empty:
                        dataframe = dataframe[required_columns].copy(); dataframe.dropna(inplace=True)
                        try: 
                            dataframe['Slide'] = pd.to_numeric(dataframe['Slide'], errors='coerce').astype('Int64')
                            dataframe.dropna(subset=['Slide'], inplace=True)
                        except Exception: return None
                        dataframe['Original'] = dataframe['Original'].astype(str); dataframe['Encripted'] = dataframe['Encripted'].astype(str)
                        if dataframe.empty: return None
                    else: return None
                except Exception: return None
            elif isinstance(source, pd.DataFrame):
                 required_columns = ['Original', 'Slide', 'Encripted']
                 if all(col in source.columns for col in required_columns):
                     dataframe = source[required_columns].copy(); dataframe.dropna(inplace=True)
                     try: 
                         dataframe['Slide'] = pd.to_numeric(dataframe['Slide'], errors='coerce').astype('Int64')
                         dataframe.dropna(subset=['Slide'], inplace=True)
                     except Exception: return None
                     dataframe['Original'] = dataframe['Original'].astype(str); dataframe['Encripted'] = dataframe['Encripted'].astype(str)
                     if dataframe.empty: return None
                 else:
                     print(f"Помилка: Наданий DataFrame не містить ({', '.join(required_columns)}).")
                     return None
            else:
                 print("Помилка ініціалізації: Непідтримуваний тип 'source'.")
                 return None
        return dataframe

    def _train_model(self, df: pd.DataFrame):
        """
        Навчає модель, заповнюючи словники відповідностей символів
        для кожного зсуву (1, 2, 3, 4), знайденого в DataFrame.
        """
        print("Навчання моделі Цезаря: збір відповідностей для кожного зсуву...")
        total_pairs_learned = 0
        valid_shifts_found = set()

        for _, row in df.iterrows():
            original_word = str(row['Original']).lower()
            encrypted_word = str(row['Encripted']).lower()
            shift = row['Slide'] 

            if pd.isna(shift) or shift not in self.EXPECTED_SHIFTS:
                # print(f"Попередження: Пропуск рядка з несподіваним зсувом: {shift}")
                continue
            if len(original_word) != len(encrypted_word):
                # print(f"Попередження: Пропуск рядка з різною довжиною: '{original_word}', '{encrypted_word}'")
                continue

            valid_shifts_found.add(shift)
            current_mapping = self.learned_mappings.setdefault(shift, {})

            # Заповнюємо мапу для цього зсуву
            for i in range(len(original_word)):
                orig_char = original_word[i]
                enc_char = encrypted_word[i]

                if 'a' <= enc_char <= 'z' and 'a' <= orig_char <= 'z':
                    if enc_char in current_mapping and current_mapping[enc_char] != orig_char:
                        print(f"!!! Попередження: Конфлікт даних для зсуву {shift}! "
                              f"'{enc_char}' -> '{current_mapping[enc_char]}' vs '{orig_char}'. "
                              f"Перезаписуємо новим значенням.")
                    current_mapping[enc_char] = orig_char
                    total_pairs_learned += 1 

        print("Навчання завершено.")
        if not self.learned_mappings:
             print("Модель порожня. Не знайдено валідних даних для навчання.")
        else:
             print(f"Вивчено відповідності для зсувів: {sorted(list(self.learned_mappings.keys()))}")
             # for s, m in self.learned_mappings.items():
             #     print(f"  Зсув {s}: вивчено {len(m)} пар символів.")

    def decrypt(self, encrypted_text: str, shift: int) -> str:
        decrypted_text = ""

        if shift not in self.learned_mappings:
            print(f"Попередження: Модель не навчалася для зсуву {shift}. Повернення оригінального тексту.")
            return encrypted_text

        current_mapping = self.learned_mappings[shift]

        for char in encrypted_text:
            decrypted_char = None 

            if 'a' <= char.lower() <= 'z':
                lower_char = char.lower()

                predicted_original_char = current_mapping.get(lower_char)

                if predicted_original_char:
                    decrypted_char = predicted_original_char.upper() if char.isupper() else predicted_original_char
                else:
                    decrypted_char = char
            else:
                decrypted_char = char 

            decrypted_text += decrypted_char

        return decrypted_text

# --- Приклад використання ---
if __name__ == "__main__":

    print("--- Тест 1: Ініціалізація з РЯДКА CSV ---")

    temp_file_path = "D:/Mysor2/3kurs/Coll_proc_data/Project/NewGen/data_r.csv"

    caesar_decryptor = CaesarLearnedDecryptor(temp_file_path, is_file_path=True)
    print("\n--- Тестування дешифрування (ТІЛЬКИ на основі даних CSV) ---")

    text_s1 = "Bmxbzt dpvout tnppuit. B Z A."
    print(f"Зсув 1: '{text_s1}' -> '{caesar_decryptor.decrypt(text_s1, 1)}'") # Очікуємо: Always counts smooths. A Y Z.

    text_s2 = "Cnycau eqwpvu uoqqvju. D Y A."
    print(f"Зсув 2: '{text_s2}' -> '{caesar_decryptor.decrypt(text_s2, 2)}'") # Очікуємо: Always counts smooths. B W Y.

    text_s3 = "Dozdbv frxqw vprrwkv. F X A."
    print(f"Зсув 3: '{text_s3}' -> '{caesar_decryptor.decrypt(text_s3, 3)}'") # Очікуємо: Always count smooths. C U X.

    text_s4 = "Epaecw wqssxlw gsyrx. H W A."
    print(f"Зсув 4: '{text_s4}' -> '{caesar_decryptor.decrypt(text_s4, 4)}'") # Очікуємо: Always smooths count. D S W.

    text_s5 = "Xzhp inhp"
    print(f"Зсув 5: '{text_s5}' -> '{caesar_decryptor.decrypt(text_s5, 5)}'") # Очікуємо попередження і повернення оригінального тексту

    print("\n--- Тест 2: Ініціалізація з порожніми даними ---")
    empty_decryptor = CaesarLearnedDecryptor("", is_file_path=False)
    print(f"Зсув 1 (порожня модель): '{text_s1}' -> '{empty_decryptor.decrypt(text_s1, 1)}'") # Очікуємо повернення оригінального тексту
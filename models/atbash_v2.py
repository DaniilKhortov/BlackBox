import pandas as pd
import io
import string
import os
from typing import Dict, Optional, Union, List
from collections import Counter

def load_words_from_csv(file_path: str) -> Optional[pd.DataFrame]:

    try:
        df = pd.read_csv(file_path)
        if df.empty: return None
        required_columns = ['Original', 'Encripted']
        if not all(col in df.columns for col in required_columns):
            print(f"Помилка: У файлі '{file_path}' відсутні необхідні колонки ({', '.join(required_columns)}).")
            return None
        df_selected = df[required_columns].copy()
        df_selected.dropna(inplace=True)
        df_selected['Original'] = df_selected['Original'].astype(str)
        df_selected['Encripted'] = df_selected['Encripted'].astype(str)
        if df_selected.empty: return None
        return df_selected
    except FileNotFoundError:
        print(f"Помилка: Файл '{file_path}' не знайдено.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Помилка: Файл '{file_path}' порожній або має неправильний формат.")
        return None
    except Exception as e:
        print(f"Сталася неочікувана помилка під час читання файлу '{file_path}': {e}")
        return None

# --- Клас Дешифратора ---
class AtbashLearnedOnlyDecryptor: # Змінив назву для ясності


    def __init__(self, source: Union[str, pd.DataFrame], is_file_path: bool = False):
        """
        Ініціалізує дешифратор та "навчає" його на основі CSV,
        зберігаючи спостережувані пари символів.
        """
        self.learned_pairs: Dict[str, List[str]] = {char: [] for char in string.ascii_lowercase}


        dataframe = self._load_dataframe(source, is_file_path)

        if dataframe is None:
             print("Попередження: Не вдалося завантажити дані для навчання. Модель буде порожньою.")
             # Залишаємо learned_pairs порожнім
        else:
            self._train_model(dataframe) # Навчаємо лише якщо дані є

    # _load_dataframe (без змін)
    def _load_dataframe(self, source: Union[str, pd.DataFrame], is_file_path: bool) -> Optional[pd.DataFrame]:
        dataframe = None
        if is_file_path:
            if isinstance(source, str):
                dataframe = load_words_from_csv(source)
            else:
                 print("Помилка ініціалізації: Якщо is_file_path=True, 'source' має бути рядком.")
        else:
            if isinstance(source, str):
                try:
                    dataframe = pd.read_csv(io.StringIO(source))
                    required_columns = ['Original', 'Encripted']
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
                 else: print(f"Помилка: Наданий DataFrame не містить ({', '.join(required_columns)})."); return None
            else: print("Помилка ініціалізації: Непідтримуваний тип 'source'."); return None
        return dataframe

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
        """
        "Передбачає" оригінальний символ ТІЛЬКИ на основі навчених даних.
        """
        if enc_char_lower in self.learned_pairs and self.learned_pairs[enc_char_lower]:
            most_common_pair = Counter(self.learned_pairs[enc_char_lower]).most_common(1)
            if most_common_pair:
                return most_common_pair[0][0]
        return None # Повертає None, якщо даних немає

    def decrypt(self, encrypted_text: str) -> str:
        """
        Дешифрує текст, використовуючи ТІЛЬКИ "передбачення" моделі,
        навченої на CSV даних
        """
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

# --- Приклад використання  ---
if __name__ == "__main__":
    
    temp_file_path = "D:/Mysor2/3kurs/Coll_proc_data/Project/NewGen/EnigmaticCodes.Atbash.csv"

    print("\n--- Тест 1: Ініціалізація з ШЛЯХУ до файлу (Новий клас) ---")
    decryptor_from_file = AtbashLearnedOnlyDecryptor(temp_file_path, is_file_path=True)
    print("\nДешифрування ТІЛЬКИ на основі даних файлу:")

    test_text1 = "nfg hzw svool nrhgzpv"
    print(f"'{test_text1}' -> '{decryptor_from_file.decrypt(test_text1)}'")


    test_text2 = "Gsv dliow Zzz"
    print(f"'{test_text2}' -> '{decryptor_from_file.decrypt(test_text2)}'")

    test_text3 = "xsvxp lmxv zmw gdrxv ru blf ziv ivzwrmt gsrh blf hfxxvvwvw"
    print(f"'{test_text3}' -> '{decryptor_from_file.decrypt(test_text3)}'")

    print("\n--- Тест 2: Ініціалізація з РЯДКА CSV (Новий клас) ---")
    csv_string_data = """Original,Encripted
python,kbgslm
code,xlwv
""" 
    decryptor_from_string = AtbashLearnedOnlyDecryptor(csv_string_data, is_file_path=False)
    print("\nДешифрування ТІЛЬКИ на основі даних рядка:")
    print(f"'kbgslm xlwv' -> '{decryptor_from_string.decrypt('kbgslm xlwv')}'") # Має бути 'python code'
    print(f"'Svool' -> '{decryptor_from_string.decrypt('Svool')}'") # Очікуємо 'Svool' або схоже

    print("\n--- Тест 4: Ініціалізація з некоректним шляхом (Новий клас) ---")
    decryptor_bad_path = AtbashLearnedOnlyDecryptor("non_existent_file.csv", is_file_path=True)
    print("\nДешифрування (некоректний шлях, модель порожня):")
    print(f"'gsv dliow kbgslm' -> '{decryptor_bad_path.decrypt('gsv dliow kbgslm')}'") # Очікуємо 'gsv dliow kbgslm'
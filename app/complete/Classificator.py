from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import string
from typing import Tuple, List, Dict, Optional
import re 
import os
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_CHARS_PATTERN = re.compile(r"^[a-zA-Z0-9\s" + re.escape(string.punctuation) + r"]*$")

try:
    from .Atbash import AtbashDecryptor # Повертає str
    from .Caesar import CaesarDecryptor # Повертає Tuple[Optional[int], str]
    from .Pl import PigLatinDecryptor     # Повертає str
    DECRYPTORS_IMPORTED = True
except ImportError as e:
    # Зберігаємо повідомлення про помилку імпорту
    INITIAL_IMPORT_ERROR = f"ПОМИЛКА імпорту дешифраторів: {e}. Дешифрування буде неможливе."
    print(INITIAL_IMPORT_ERROR) # Виводимо одразу для ясності
    AtbashDecryptor = None
    CaesarDecryptor = None
    PigLatinDecryptor = None
    DECRYPTORS_IMPORTED = False
else:
    INITIAL_IMPORT_ERROR = None # Немає помилки імпорту
# Контекстні менеджери для перехоплення виводу
import io
import contextlib
#Завантаження алфавіту
alphabet = string.ascii_lowercase
vowels = set("aeiou")

#Переведення літер до чисел. Маємо словник (як і справжній словник!!!) з літерами та їх позицією в алфавіті
charToInt = {char: i + 1 for i, char in enumerate(alphabet)}


#Завантаження словників слів з БД
def loadData():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["EnigmaticCodes"]
    data = []
    
    #Перенесення шифрованих слів з колекції у масив з tuple (Слово, шифр). 
    for method, label in [('Atbash_real', 'atbash'), ('Caesar_real', 'caesar'), ('PL_real', 'pl')]:
        collection = db[method]
        for doc in collection.find({}, {'Encripted': 1, '_id': 0}):
            word = doc['Encripted']
            data.append((word, label))
            
    return data


def extractFeatures(word, maxLen=12):
    #Перепис слова у масив цифр. Кожна цифра - позиція літери слова
    encoded = [charToInt.get(c, 0) for c in word[:maxLen]]
    
    #Представлення слова у вигляді ячейки до 12 символів. Пропуски заповнюються 0
    padded = encoded + [0] * (maxLen - len(encoded))  
    
    #Частота кожної літери
    letterFrequency = [word.count(c) / len(word) for c in alphabet]
    
    #Числові ознаки слова. Оскільки слово - це набір літер, то аналіз відбувається й математично
    letters = [charToInt.get(c, 0) for c in word if c in alphabet]  
    mean = np.mean(letters) if letters else 0                   #Середнє квадратичне відхилення
    std = np.std(letters) if letters else 0                     #Стандартне відхилення
    minLetter = min(letters) if letters else 0                   #Найменша літера (Найперша за порядком алфавіту літера слова)
    maxLetter = max(letters) if letters else 0                   #Найостанніша літера (Найостанніша за порядком алфавіту літера слова)
    vowelRatio = sum(1 for c in word if c in vowels) / len(word) #Частка голосних літер у слові
    
    return padded + [mean, std, minLetter, maxLetter, vowelRatio] + letterFrequency 

#Метод прогнозу шифру слова
def predictMethod(word):
    #Розбір на ознаки
    features = extractFeatures(word)
    pred = clf.predict([features])[0]
    #Перекодування класу з числа до мітки
    return le.inverse_transform([pred])[0]

#Конвертація слова
def prepareData(sentence):
    words = sentence.split()
    output = []
    for word in words:
        word = word.lower()
        word = word.replace('\u2060', '').replace(';', '').replace('…', '').replace('ü', 'u').replace('ö', 'u').replace('!', '').replace(',', '').replace('?', '').replace('‘', '').replace('”', '').replace('—', '').replace('-', '').replace('“', '').replace('ä', 'a').replace('’', '').replace(':', '').replace(',', '')
        word = word.strip(string.punctuation)  
        output.append(word)
    return output 


#Визначення шифру речення
def mostFrequent(List):
    classes = {
        "atbash":0,
        "caesar":0,
        "pl":0
    }
    
    for n in List:
        if n in classes:
            classes[n] += 1 
             
    print(classes)     
    return max(classes, key=classes.get)

clf: Optional[RandomForestClassifier] = None
le: Optional[LabelEncoder] = None
enigma_a: Optional[AtbashDecryptor] = None
enigma_c: Optional[CaesarDecryptor] = None
enigma_pl: Optional[PigLatinDecryptor] = None
is_initialized = False
initialization_messages: List[str] = []

# --- Функція ініціалізації (збирає повідомлення) ---
def initialize_system():
    global clf, le, enigma_a, enigma_c, enigma_pl, is_initialized, initialization_messages
    if is_initialized:
        # Не додаємо повідомлення тут, бо воно додається в інтерфейсній функції
        return

    # Очищуємо попередні повідомлення, якщо викликаємо повторно (хоча не мали б)
    initialization_messages.clear()
    if INITIAL_IMPORT_ERROR: initialization_messages.append(INITIAL_IMPORT_ERROR)

    #initialization_messages.append("ІНФО: Запуск ініціалізації системи...")

    # 1. Навчання Класифікатора
    training_data = loadData()
    if not training_data:
        initialization_messages.append("КРИТИЧНО: Немає даних для навчання класифікатора. Ініціалізація зупинена.")
        return

    try:
        # Використовуємо list comprehension з перевіркою, щоб уникнути помилок на великих даних
        features_list = []
        labels_list = []
        invalid_feature_count = 0
        for word, label in training_data:
             features = extractFeatures(word)
             # Перевірка розмірності (має бути константною)
             if features and len(features) == (12 + 5 + len(alphabet)):
                 features_list.append(features)
                 labels_list.append(label)
             else:
                 invalid_feature_count += 1
        if invalid_feature_count > 0:
             initialization_messages.append(f"ПОПЕРЕДЖЕННЯ: Пропущено {invalid_feature_count} слів через некоректні ознаки.")
        if not features_list:
             initialization_messages.append("КРИТИЧНО: Не вдалося виділити ознаки для жодного слова. Ініціалізація зупинена.")
             return

        x = np.array(features_list)
        y = np.array(labels_list)

    except Exception as e:
        initialization_messages.append(f"КРИТИЧНО: Помилка під час формування масивів ознак/міток: {e}. Ініціалізація зупинена.")
        return

    le = LabelEncoder()
    try:
        yEncoded = le.fit_transform(y)
        #initialization_messages.append(f"ІНФО: Мітки класів закодовано: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    except Exception as e:
        initialization_messages.append(f"КРИТИЧНО: Помилка кодування міток: {e}. Ініціалізація зупинена.")
        le = None; return

    try:
        # Перевірка, чи достатньо даних для розділення
        if len(np.unique(yEncoded)) < 2 or x.shape[0] < 2:
             initialization_messages.append("ПОПЕРЕДЖЕННЯ: Недостатньо даних або класів для розділення на train/test. Навчання на всіх даних.")
             xTrain, yTrain = x, yEncoded
             test_acc_msg = "Оцінка точності не проводилась."
        else:
             test_size = 0.2
             # Stratify може викликати помилку, якщо класів мало
             try:
                 xTrain, xTest, yTrain, yTest = train_test_split(x, yEncoded, test_size=test_size, random_state=42, stratify=yEncoded)
             except ValueError:
                 initialization_messages.append("ПОПЕРЕДЖЕННЯ: Не вдалося виконати стратифіковане розділення, виконується звичайне.")
                 xTrain, xTest, yTrain, yTest = train_test_split(x, yEncoded, test_size=test_size, random_state=42)

             clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
             clf.fit(xTrain, yTrain)
             acc = clf.score(xTest, yTest)
             test_acc_msg = f"Точність моделі класифікації шифрів (на тестових даних): {acc*100:.2f}%"

        # Остаточне навчання на всіх даних (поширена практика після оцінки)
        clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
        clf.fit(x, yEncoded) # Навчаємо на всіх даних
        #initialization_messages.append(f"ІНФО: Класифікатор навчено на {x.shape[0]} прикладах. {test_acc_msg}")

    except Exception as e:
        initialization_messages.append(f"КРИТИЧНО: Помилка тренування/оцінки класифікатора: {e}. Ініціалізація зупинена.")
        clf = None; return

    # 2. Ініціалізація Дешифраторів
    DB_NAME = "EnigmaticCodes"
    #initialization_messages.append("ІНФО: Ініціалізація дешифраторів з MongoDB...")
    init_decryptor_errors = False

    if DECRYPTORS_IMPORTED:
        # Функція-хелпер для ініціалізації
        def init_decryptor(DecryptorClass, name, collection):
            nonlocal init_decryptor_errors
            decryptor = None
            msg_list = []
            if DecryptorClass:
                try:
                    # Використовуємо contextlib для перехоплення можливого print всередині конструктора
                    stdout_capture = io.StringIO()
                    with contextlib.redirect_stdout(stdout_capture):
                         decryptor = DecryptorClass(DB_NAME, collection)
                    captured_stdout = stdout_capture.getvalue().strip()
                    if captured_stdout: msg_list.append(f"  stdout({name}): {captured_stdout}")

                    if decryptor: # Якщо конструктор успішно повернув об'єкт
                        pass
                        #msg_list.append(f"ІНФО: Дешифратор {name} ініціалізовано.")
                    else: # Якщо конструктор повернув None або виникла помилка всередині
                         #msg_list.append(f"ПОМИЛКА: Ініціалізація {name} не вдалася (конструктор не повернув об'єкт).")
                         init_decryptor_errors = True
                except Exception as init_e:
                    msg_list.append(f"ПОМИЛКА: Неочікувана помилка ініціалізації {name}: {init_e}")
                    init_decryptor_errors = True
            else:
                msg_list.append(f"ПОПЕРЕДЖЕННЯ: Клас {name} не імпортовано.")
                init_decryptor_errors = True # Вважаємо помилкою, якщо мав бути
            return decryptor, msg_list

        enigma_a, msgs_a = init_decryptor(AtbashDecryptor, "Atbash", 'Atbash_real')
        initialization_messages.extend(msgs_a)
        enigma_c, msgs_c = init_decryptor(CaesarDecryptor, "Caesar", 'Caesar_real')
        initialization_messages.extend(msgs_c)
        enigma_pl, msgs_pl = init_decryptor(PigLatinDecryptor, "Pig Latin", 'PL_real')
        initialization_messages.extend(msgs_pl)

    else: # Якщо імпорт не вдався
        initialization_messages.append("ПОМИЛКА: Дешифратори не були імпортовані. Ініціалізація дешифраторів пропущена.")
        init_decryptor_errors = True

    # Ставимо is_initialized=True, тільки якщо ВСЕ необхідне готове
    if clf and le and DECRYPTORS_IMPORTED and not init_decryptor_errors:
        is_initialized = True
        #initialization_messages.append("ІНФО: Ініціалізація системи успішно завершена.")
    else:
         initialization_messages.append("ПОМИЛКА: Ініціалізація системи НЕ завершена через помилки.")


# --- Функція прогнозу для одного слова (без змін, але з перевіркою clf/le) ---
def predictMethod(word: str) -> Optional[str]:
    if not clf or not le: return None # Перевірка наявності моделей
    try:
        features = extractFeatures(word)
        if not features or len(features) != clf.n_features_in_: return None
        features_array = np.array(features).reshape(1, -1)
        pred_encoded = clf.predict(features_array)[0]
        pred_label = le.inverse_transform([pred_encoded])[0]
        return pred_label
    except Exception: return None


# --- ОСНОВНА ФУНКЦІЯ-ІНТЕРФЕЙС (повертає помилки/попередження) ---
def decrypt_sentence_interface(encrypted_sentence: str) -> Tuple[Optional[str], str, List[str]]:
    """
    Визначає тип шифру, дешифрує та повертає результат разом з повідомленнями.
    Виконує валідацію вхідного тексту.
    """
    messages: List[str] = []

    # 1. Перевірка ініціалізації
    if not is_initialized:
        #messages.append("ІНФО: Спроба ініціалізації системи...")
        # Очищуємо глобальні повідомлення перед новим викликом
        initialization_messages.clear()
        if INITIAL_IMPORT_ERROR: initialization_messages.append(INITIAL_IMPORT_ERROR)
        initialize_system()
        messages.extend(initialization_messages) # Додаємо всі повідомлення ініціалізації
        if not is_initialized:
            messages.append("ПОМИЛКА: Не вдалося ініціалізувати систему. Дешифрування неможливе.")
            # Повертаємо None як тип, оригінальне речення і зібрані повідомлення
            return None, encrypted_sentence, messages

    #messages.append(f"ІНФО: Отримано речення: '{encrypted_sentence}'")

    # 2. Валідація вхідного речення
    if not isinstance(encrypted_sentence, str) or not encrypted_sentence.strip():
        messages.append("ПОМИЛКА: Вхідне речення порожнє або не є рядком.")
        return None, encrypted_sentence, messages
    if not ALLOWED_CHARS_PATTERN.match(encrypted_sentence):
        messages.append("ПОМИЛКА: Вхідне речення містить недозволені символи.")
        return None, encrypted_sentence, messages

    # 3. Підготовка вхідних даних
    input_words = prepareData(encrypted_sentence)
    if not input_words:
        messages.append("ПОПЕРЕДЖЕННЯ: Не знайдено слів для аналізу після очищення речення.")
        return None, encrypted_sentence, messages

    # 4. Прогнозування класу для кожного слова
    predicted_classes = []
    #messages.append("ІНФО: Прогнозування типу шифру для слів:")
    prediction_errors = 0
    for word in input_words:
        predicted_class = predictMethod(word) # Вже обробляє помилки всередині
        if predicted_class:
            predicted_classes.append(predicted_class)
            #messages.append(f"  Слово: '{word}' -> Прогноз: {predicted_class}")
        else:
            messages.append(f"ПОПЕРЕДЖЕННЯ: Не вдалося спрогнозувати тип для слова '{word}'.")
            prediction_errors += 1

    if not predicted_classes:
        messages.append("ПОМИЛКА: Не вдалося визначити тип шифру для жодного слова.")
        return None, encrypted_sentence, messages
    if prediction_errors > 0:
         messages.append(f"ПОПЕРЕДЖЕННЯ: Не вдалося спрогнозувати тип для {prediction_errors} слів.")

    # 5. Визначення основного шифру речення
    sentence_class = mostFrequent(predicted_classes)
    #messages.append(f"ІНФО: Найімовірніший тип шифру речення: {sentence_class}")

    # 6. Дешифрування
    decrypted_sentence = encrypted_sentence
    detected_cipher_type = sentence_class

    try:
        decryptor_instance = None
        if sentence_class == "atbash":
            decryptor_instance = enigma_a
            if decryptor_instance:
                decrypted_sentence = decryptor_instance.decrypt(encrypted_sentence)
                #messages.append("ІНФО: Застосовано дешифратор Atbash.")
            else: messages.append("ПОМИЛКА: Дешифратор Atbash не ініціалізований!")
        elif sentence_class == "caesar":
            decryptor_instance = enigma_c
            if decryptor_instance:
                identified_shift, decrypted_text_c = decryptor_instance.decrypt(encrypted_sentence)
                if identified_shift is not None:
                    decrypted_sentence = decrypted_text_c
                   # messages.append(f"ІНФО: Застосовано дешифратор Caesar (зсув: {identified_shift}).")
                else:
                     messages.append("ПОПЕРЕДЖЕННЯ: Дешифратор Caesar не зміг визначити зсув.")
                     detected_cipher_type = None # Вважаємо, що дешифрування не вдалося
            else: messages.append("ПОМИЛКА: Дешифратор Caesar не ініціалізований!")
        elif sentence_class == "pl":
            decryptor_instance = enigma_pl
            if decryptor_instance:
                decrypted_sentence = decryptor_instance.decrypt(encrypted_sentence)
                #messages.append("ІНФО: Застосовано дешифратор Pig Latin.")
            else: messages.append("ПОМИЛКА: Дешифратор Pig Latin не ініціалізований!")
        else:
            messages.append(f"ПОПЕРЕДЖЕННЯ: Невідомий тип шифру '{sentence_class}'.")
            detected_cipher_type = None

        # Якщо відповідний дешифратор не був ініціалізований, скидаємо тип
        if not decryptor_instance and sentence_class in ["atbash", "caesar", "pl"]:
            detected_cipher_type = None

    except Exception as e:
        messages.append(f"КРИТИЧНО: Помилка під час виклику .decrypt() для '{sentence_class}': {e}")
        decrypted_sentence = encrypted_sentence # Повертаємо оригінал
        detected_cipher_type = None

    # Якщо тип скинуто через помилку, повертаємо None
    final_type = detected_cipher_type if (sentence_class == detected_cipher_type) else None

    return final_type, decrypted_sentence, messages

    
    
    
    

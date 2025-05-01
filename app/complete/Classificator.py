from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import string
from typing import Tuple, List, Dict, Optional

from Atbash import AtbashDecryptor 
from Caesar import CaesarDecryptor
from Pl import PigLatinDecryptor

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

def initialize_system():
    """Завантажує дані, тренує класифікатор та ініціалізує дешифратори."""
    global clf, le, enigma_a, enigma_c, enigma_pl, is_initialized
    if is_initialized:
        print("Система вже ініціалізована.")
        return

    #print("Запуск ініціалізації системи...")

    training_data = loadData()
    if not training_data:
        print("ПОМИЛКА: Немає даних для навчання класифікатора. Ініціалізація неможлива.")
        return

    try:
        x = np.array([extractFeatures(word) for word, _ in training_data])
        y = np.array([label for _, label in training_data])
    except Exception as e:
        print(f"ПОМИЛКА під час виділення ознак: {e}")
        return

    le = LabelEncoder()
    try:
        yEncoded = le.fit_transform(y)
    except Exception as e:
        print(f"ПОМИЛКА під час кодування міток: {e}")
        le = None 
        return

    try:
        xTrain, xTest, yTrain, yTest = train_test_split(x, yEncoded, test_size=0.2, random_state=42, stratify=yEncoded)
        clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced') # Збільшив дерева, додав балансування
        clf.fit(xTrain, yTrain)
        acc = clf.score(xTest, yTest)
        print(f"Точність моделі класифікації шифрів: {acc*100:.2f}%")
    except Exception as e:
        print(f"ПОМИЛКА під час тренування або оцінки класифікатора: {e}")
        clf = None 
        return

    DB_NAME = "EnigmaticCodes"
    #print("Ініціалізація дешифраторів з MongoDB...")
    try:
        if AtbashDecryptor:
             enigma_a = AtbashDecryptor(DB_NAME, 'Atbash_real')
        if CaesarDecryptor:
             enigma_c = CaesarDecryptor(DB_NAME, 'Caesar_real')
        if PigLatinDecryptor:
             enigma_pl = PigLatinDecryptor(DB_NAME, 'PL_real')
    except Exception as e:
         print(f"ПОМИЛКА під час ініціалізації дешифраторів: {e}")

    is_initialized = True
    #print("Ініціалізація системи завершена.")

def decrypt_sentence_interface(encrypted_sentence: str) -> Tuple[Optional[str], str]:
    if not is_initialized:
        print("Спроба ініціалізації системи...")
        initialize_system()
        if not is_initialized:
            print("ПОМИЛКА: Не вдалося ініціалізувати систему.")
            return None, encrypted_sentence

    print(f"\nОтримано речення для дешифрування: '{encrypted_sentence}'")

    input_words = prepareData(encrypted_sentence)
    if not input_words:
        print("Попередження: Не знайдено слів для аналізу в реченні.")
        return None, encrypted_sentence

    predicted_classes = []
    #print("Прогнозування типу шифру для слів:")
    for word in input_words:
        predicted_class = predictMethod(word)
        if predicted_class:
            predicted_classes.append(predicted_class)
            #print(f"  '{word}' -> {predicted_class}")
        else:
            print(f"  '{word}' -> Не вдалося спрогнозувати")


    if not predicted_classes:
        print("Попередження: Не вдалося визначити тип шифру для жодного слова.")
        return None, encrypted_sentence

    sentence_class = mostFrequent(predicted_classes)
    #print(f"Найімовірніший тип шифру речення: {sentence_class}")

    decrypted_sentence = encrypted_sentence
    detected_cipher_type = sentence_class   

    try:
        if sentence_class == "atbash":
            if enigma_a:
                decrypted_sentence = enigma_a.decrypt(encrypted_sentence)
                print("Застосовано дешифратор Atbash.")
            else:
                print("ПОМИЛКА: Дешифратор Atbash не ініціалізований!")
                detected_cipher_type = None # Помилка, тип не застосовано
        elif sentence_class == "caesar":
            if enigma_c:
                # Метод decrypt для Caesar повертає (shift, text)
                identified_shift, decrypted_text_c = enigma_c.decrypt(encrypted_sentence)
                if identified_shift is not None:
                    decrypted_sentence = decrypted_text_c
                    print(f"Застосовано дешифратор Caesar (визначено зсув: {identified_shift}).")
                else:
                     print("Попередження: Дешифратор Caesar не зміг визначити зсув, повертається оригінал.")
                     detected_cipher_type = None 
            else:
                print("ПОМИЛКА: Дешифратор Caesar не ініціалізований!")
                detected_cipher_type = None
        elif sentence_class == "pl":
            if enigma_pl:
                decrypted_sentence = enigma_pl.decrypt(encrypted_sentence)
                print("Застосовано дешифратор Pig Latin.")
            else:
                print("ПОМИЛКА: Дешифратор Pig Latin не ініціалізований!")
                detected_cipher_type = None
        else:
            print(f"Попередження: Невідомий або непідтримуваний клас шифру '{sentence_class}'.")
            detected_cipher_type = None 

    except Exception as e:
        print(f"ПОМИЛКА під час виклику методу decrypt для '{sentence_class}': {e}")
        decrypted_sentence = encrypted_sentence 
        detected_cipher_type = None

    return detected_cipher_type, decrypted_sentence

    
    
    
    

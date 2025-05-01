from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import string

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
    for method, label in [('Atbash', 'atbash'), ('Caesar', 'caesar'), ('PL', 'pl')]:
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

#Завантаження словника
data = loadData()
#Х - це розібрані на ознаки слова словника
x = np.array([extractFeatures(word) for word, _ in data])

#У - це маска з належностей слів словника
y = np.array([label for _, label in data])

#Заміна типів шифрів У на числв. 0-Атбаш, 1-Цезар, 2-Свиняча латина
le = LabelEncoder()
yEncoded = le.fit_transform(y)

#Тренування моделі Random Forest
xTrain, xTest, yTrain, yTest = train_test_split(x, yEncoded, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42) #Встановлюємо 100 дерев
clf.fit(xTrain, yTrain)

#Точність тестових данних
acc = clf.score(xTest, yTest)
print(f"Точність моделі: {acc:.2f}%",)


inputSentence = "Pm ol ohk hufaopun jvumpkluaphs av zhf, ol dyval pa pu jpwoly, aoha pz, if zv johunpun aol vykly vm aol slaalyz vm aol hswohila, aoha uva h dvyk jvbsk il thkl vba."
inputData = prepareData(inputSentence)
inputDataClasses = []

for word in inputData:
    wordClass = predictMethod(word)
    print(f"Метод шифрування слова {word}: ", wordClass)
    inputDataClasses.append(wordClass)


print("Метод шифрування речення:"+mostFrequent(inputDataClasses))    
    
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import string

# Параметри
alphabet = string.ascii_lowercase
vowels = set("aeiou")
char_to_int = {char: i + 1 for i, char in enumerate(alphabet)}  # a=1 ... z=26

# Функція ознак
def extract_features(word, max_len=12):
    word = word.lower()
    encoded = [char_to_int.get(c, 0) for c in word[:max_len]]
    padded = encoded + [0] * (max_len - len(encoded))

    # Частотний вектор
    freq = [word.count(c) / len(word) for c in alphabet]

    # Числові ознаки
    letters = [char_to_int.get(c, 0) for c in word if c in alphabet]
    mean = np.mean(letters) if letters else 0
    std = np.std(letters) if letters else 0
    min_char = min(letters) if letters else 0
    max_char = max(letters) if letters else 0
    vowel_ratio = sum(1 for c in word if c in vowels) / len(word)

    return padded + [mean, std, min_char, max_char, vowel_ratio] + freq

# Завантаження з MongoDB
def load_data():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["EnigmaticCodes"]
    data = []
    
    for method, label in [('Atbash', 'atbash'), ('Caesar', 'caesar'), ('PL', 'pl')]:
        collection = db[method]
        for doc in collection.find({}, {'Encripted': 1, '_id': 0}):
            word = doc['Encripted']
            data.append((word, label))
    return data

# Формуємо X та y
data = load_data()
X = np.array([extract_features(word) for word, _ in data])
y = np.array([label for _, label in data])
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Тренування
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Точність
acc = clf.score(X_test, y_test)
print(f"📊 Точність з додатковими ознаками: {acc:.2f}")

# Функція передбачення
def predict_method(word):
    features = extract_features(word)
    pred = clf.predict([features])[0]
    return le.inverse_transform([pred])[0]

# Приклад
example = "ellohay"
print("Метод шифрування:", predict_method(example))

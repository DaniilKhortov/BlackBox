from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import string

# 1. З'єднання з MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["EnigmaticCodes"]

# 2. Завантаження даних
def load_data():
    data = []
    
    for method, label in [('Atbash', 'atbash'), ('Caesar', 'caesar'), ('PL', 'pl')]:
        collection = db[method]
        for doc in collection.find({}, {'Encripted': 1, '_id': 0}):
            word = doc['Encripted']
            data.append((word, label))
    
    return data

# 3. Перетворення слова в вектор
alphabet = string.ascii_lowercase
char_to_int = {char: i + 1 for i, char in enumerate(alphabet)}  # a=1, b=2, ..., z=26

def encode_word(word, max_len=12):
    encoded = [char_to_int.get(c, 0) for c in word[:max_len]]
    # паддінг до фіксованої довжини
    while len(encoded) < max_len:
        encoded.append(0)
    return encoded

# 4. Формування X та y
data = load_data()
X = np.array([encode_word(word) for word, _ in data])
y = np.array([label for _, label in data])

# 5. Кодування міток
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 'atbash'→0, 'caesar'→1, 'pl'→2

# 6. Розбиття і тренування
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 7. Оцінка точності
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# 8. Передбачення прикладу
def predict_method(word):
    encoded = encode_word(word)
    pred = clf.predict([encoded])[0]
    return le.inverse_transform([pred])[0]

# Приклад
example = "yllclczovz"  # слово, зашифроване Атбашем (наприклад для "test")
print("Шифр:", predict_method(example))

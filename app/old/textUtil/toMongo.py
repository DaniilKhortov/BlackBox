from pymongo import MongoClient
from toLowerCase import wordsUnique  

def encrypt(text, s):
    result = ""
    for char in text:
        if char.isupper():
            result += chr((ord(char) + s - 65) % 26 + 65)
        else:
            result += chr((ord(char) + s - 97) % 26 + 97)
    return result


client = MongoClient("mongodb://localhost:27017/")
db = client["EnigmaticCodes"]
collection = db["Caesar"]  

s = 1
while s < 5:
    for word in wordsUnique:
        if word == "":
            continue
        encryptedWord = encrypt(word, s)
        print(f"Text  : {word}")
        print(f"Cipher: {encryptedWord}")

        mydict = {
            "Original": word,
            "Slide": s,
            "Encripted": encryptedWord
        }

        collection.insert_one(mydict)
    s += 1


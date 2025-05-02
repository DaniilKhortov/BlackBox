from pymongo import MongoClient

# Функція дешифрування шифру Цезаря
def caesar_decrypt(text, shift):
    decrypted_text = ""
    for char in text:
        if char.isalpha():  
            shift_base = ord('a') if char.islower() else ord('A')
            decrypted_text += chr((ord(char) - shift_base - shift) % 26 + shift_base)
        else:
            decrypted_text += char  
    return decrypted_text


client = MongoClient("mongodb://localhost:27017/")
db = client["EnigmaticCodes"]
collection = db["Caesar"]  


records = collection.find()

for record in records:
    original = record["Original"]
    encrypted = record["Encripted"]
    shift = record["Slide"]

    decrypted = caesar_decrypt(encrypted, shift)

    if decrypted == original:
        print(f"Коректно: {encrypted} -> {decrypted} (Shift: {shift})")
    else:
        print(f"Аномалія реверсивності: {encrypted} -> {decrypted}, має бути {original} (Shift: {shift})")

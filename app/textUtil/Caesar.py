from toLowerCase import wordsUnique
import csv
def encrypt(text,s):
    result = ""
    for i in range(len(text)):
        char = text[i]
        if (char.isupper()):
            result += chr((ord(char) + s-65) % 26 + 65)
        else:
            result += chr((ord(char) + s - 97) % 26 + 97)
    return result
s = 1
data = []

while s < 5:
    for word in wordsUnique:
        print ("Text  : " + word)
        print ("Cipher: " + encrypt(word,s))
        mydict = { "Original": word, "Slide": s, "Encripted": encrypt(word,s) }
        data.append(mydict)
    s+=1    
with open('data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["Original", "Slide", "Encripted"])
    writer.writeheader()  
    writer.writerows(data)    
    
    
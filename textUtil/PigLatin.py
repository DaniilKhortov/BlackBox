from toLowerCase import wordsUnique
import csv
import re

def toPigLatin(word):

    if re.match(r"^[a-zA-Z]+$", word):
        first_letter = word[0]
        rest_of_word = word[1:]
        if first_letter.lower() in 'aeiou':
            return word + 'yay'
        else:
            return rest_of_word + first_letter + 'ay'
    else:
        return word  # Повертаємо слово без змін, якщо воно не містить лише літери
 
data = []

for word in wordsUnique:
    print ("Text  : " + word)
    print ("Cipher: " + toPigLatin(word))
    mydict = { "Original": word, "Encripted": toPigLatin(word) }
    data.append(mydict)
    
with open('dataPL.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["Original","Encripted"])
    writer.writeheader()  
    writer.writerows(data)   
            

from wordGenerator import Random_Word
import random, csv
import re

def to_pig_latin(word):

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
i = 0

while i < 10000:
        
    text = Random_Word(random.randint(1, 10))

    print ("Text  : " + text)
    print ("Cipher: " + to_pig_latin(text))
    
    mydict = { "Original": text, "Encripted": to_pig_latin(text) }
    data.append(mydict)

        
    i+=1
    
with open('dataPL.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["Original","Encripted"])
    writer.writeheader()  
    writer.writerows(data)   

 
 


from wordGenerator import Random_Word
import random, csv


def encrypt(text,s):
    result = ""
    for i in range(len(text)):
        char = text[i]
        if (char.isupper()):
            result += chr((ord(char) + s-65) % 26 + 65)
        else:
            result += chr((ord(char) + s - 97) % 26 + 97)
    return result

i = 0
s = 1
data = []
while s < 26:
    while i < 1000:
        
        text = Random_Word(random.randint(4, 10))

        print ("Text  : " + text)
        print ("Cipher: " + encrypt(text,s))
        mydict = { "Original": text, "Slide": s, "Encripted": encrypt(text,s) }
        data.append(mydict)
        # x = mycol.insert_one(mydict)
        
        i+=1
    s+=1 
    i=0
       
with open('data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["Original", "Slide", "Encripted"])
    writer.writeheader()  
    writer.writerows(data)    
print(s)   
 
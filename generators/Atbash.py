lookup_table = {'A' : 'Z', 'B' : 'Y', 'C' : 'X', 'D' : 'W', 'E' : 'V',
        'F' : 'U', 'G' : 'T', 'H' : 'S', 'I' : 'R', 'J' : 'Q',
        'K' : 'P', 'L' : 'O', 'M' : 'N', 'N' : 'M', 'O' : 'L',
        'P' : 'K', 'Q' : 'J', 'R' : 'I', 'S' : 'H', 'T' : 'G',
        'U' : 'F', 'V' : 'E', 'W' : 'D', 'X' : 'C', 'Y' : 'B', 'Z' : 'A',
        'a' : 'z', 'b' : 'y', 'c' : 'x', 'd' : 'w', 'e' : 'v',
        'f' : 'u', 'g' : 't', 'h' : 's', 'i' : 'r', 'j' : 'q',
        'k' : 'p', 'l' : 'o', 'm' : 'n', 'n' : 'm', 'o' : 'l',
        'p' : 'k', 'q' : 'j', 'r' : 'i', 's' : 'h', 't' : 'g',
        'u' : 'f', 'v' : 'e', 'w' : 'd', 'x' : 'c', 'y' : 'b', 'z' : 'a'}

from wordGenerator import Random_Word
import random, csv
 
def atbash(message):
    cipher = ''
    for letter in message:

        if(letter != ' '):

            cipher += lookup_table[letter]
        else:

            cipher += ' '
 
    return cipher
 
data = []
i = 0

while i < 10000:
        
    text = Random_Word(random.randint(1, 10))

    print ("Text  : " + text)
    print ("Cipher: " + atbash(text))
    
    mydict = { "Original": text, "Encripted": atbash(text) }
    data.append(mydict)

        
    i+=1
    
with open('dataA.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["Original","Encripted"])
    writer.writeheader()  
    writer.writerows(data)   

 
 


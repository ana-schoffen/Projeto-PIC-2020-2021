from nltk.tokenize import regexp_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import json
def le_json():
    with open("padrao_ouroOK.json", "r") as json_file:    
        lista = json.load(json_file)
    return lista 
def separa_c_s(lista):
    lista_c = []
    lista_s = []

    x = len(lista)

    for i in range(x):
        y = len(lista[i])
        for j in range(y):
            lista_c.append(lista[i][j][0])
            lista_s.append(lista[i][j][1])
    return lista_c, lista_s

lista = le_json()
lista_c, lista_s = separa_c_s(lista)

tam = len(lista_s)
tokens = []
num_tokens = []

for i in range(tam):
    aux = word_tokenize(lista_s[i])
    tokens.append(aux)
    num_tokens.append(len(aux))




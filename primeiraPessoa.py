import json
import re

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
ppyn = []

for i in range(tam):
    if re.search('\\beu\\b', lista_s[i], re.IGNORECASE) or re.search('\\bmeu\\b', lista_s[i], re.IGNORECASE) or re.search('\\bnós\\b', lista_s[i], re.IGNORECASE) or re.search('\\bnosso\\b', lista_s[i], re.IGNORECASE) or re.search('\\bme\\b', lista_s[i], re.IGNORECASE) or re.search('\\bmim\\b', lista_s[i], re.IGNORECASE) or re.search('\\bcomigo\\b', lista_s[i], re.IGNORECASE) or re.search('\\bminha\\b', lista_s[i], re.IGNORECASE) or re.search('\\bminhas\\b', lista_s[i], re.IGNORECASE) or re.search('\\bmeus\\b', lista_s[i], re.IGNORECASE) or re.search('\\bnossa\\b', lista_s[i], re.IGNORECASE) or re.search('\\bnossos\\b', lista_s[i], re.IGNORECASE) or re.search('\\bnossass\\b', lista_s[i], re.IGNORECASE):
        ppyn.append(1)
    else:
        ppyn.append(0)

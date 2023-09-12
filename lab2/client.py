from src.bayesian_learning import G02NaiveBayesClassifier
from src.whatsapp_regex import LOG_ENTRY_PATTERN
import re
import pdb

def data():
  FILE_PATH = './assets/chat.txt'
  PATTERN = LOG_ENTRY_PATTERN

  with open(FILE_PATH, 'r', encoding='utf-8') as file:
    data = file.read()

  matches = re.findall(PATTERN, data)
  data = [ match[1] for match in matches ]

  return data

def main():
  N = 3
  M = 1

  clf = G02NaiveBayesClassifier(data(), N=N, M=M)

  frase = []
  palabra_sugerida = ""

  print("Ingrese la frase dando ENTER luego de \x1b[3mcada palabra\x1b[0m.")
  print("Ingrese sólo ENTER para aceptar la recomendación sugerida, o escriba la siguiente palabra y de ENTER")
  print("Ingrese '.' para comenzar con una frase nueva.")
  print("Ingrese '..' para terminar el proceso.")

  while 1:
    palabra = input(">> ")

    if palabra == "..": break

    elif palabra == ".":
      print("----- Comenzando frase nueva -----")
      print(clf.update(" ".join(frase)))

      frase = []

    elif palabra == "": # acepta última palabra sugerida
      frase.append(palabra_sugerida)

    else: # escribió una palabra
      frase.append(palabra)

    if frase:
      frase_propuesta = frase.copy()
      palabra_sugerida = clf.predict(frase)
      frase_propuesta.append("\x1b[3m"+ palabra_sugerida +"\x1b[0m")

      print(" ".join(frase_propuesta))
  return 0

if __name__ == "__main__":
  main()

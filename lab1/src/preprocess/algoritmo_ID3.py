''' Algoritmo básico (notas de decisión)
Crear una raíz
• Si todos los ej. tienen el mismo valor → etiquetar con ese valor
• Si no me quedan atributos → etiquetar con el valor más común
• En caso contrario:
‣ La raíz pregunta por A, atributo que mejor clasifica los ejemplos
‣ Para cada valor vi de A
๏ Genero una rama
๏ Ejemplosvi={ejemplos en los cuales A=vi }
๏ Si Ejemplosvi es vacío → etiquetar con el valor más probable
๏ En caso contrario → ID3(Ejemplosvi, Atributos -{A})
'''

def alg_id3(data,attribute,list_attributes, min_samples_split, min_split_gain):
   

    return 0
    
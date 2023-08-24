# Metodologia para clasificacion.

Tenemos que clasificar un conjunto de datos, para eso vamos a usar un algoritmo de aprendizaje automatico.

Esta diapositiva fue introducida como parentesis para explicar como es la metodologia de trabajo a la hora de clasificar.

## Preprocesamiento de datos

Vamos a suponer que nuetro conujto de entrenamiento siempre son valores reales.

Desafortunadamente, los conjuntos de datos que obtenemos vienen de la realidad y debemos limpiarlos pq siempre hay errores.

- Lo PRIMERO que hay que hacer es partir el conjunto de datos en dos: entrenamiento y prueba.

Del conjunto de entrenamiento se usa para todo, mientras que el otro se usa para ver que tan bien le pegue.

El preprocesamiento es parte del modelado, pq involucra tomar decisiones sobre como representar la realidad a analizar.

Vimos que para atributos categoricos se puede usar:
- One hot strategy.
- Label encoding.


# Conjunto de entrenamiento, testeo y validacion.

Tenemos la data la partimos en tres, train, test y validation.
Con train armamos el modelo, con test porbamos diferentes estrategias y finalmente con validation vemos que tan bien le pega.

La idea es usar el conjunto de validacion para encontrar el punto crema.

# Estratificar segun el valor objetivo

Es una buena practica armar train test y devel de forma tal que la proporcion de valores objetivo sea la misma en los tres conjuntos.

# Cross validation

Es una tecnica para evaluar el rendimiento de un modelo. Armando varios conjuntos de train y validat e ir iterando los resultados.

# Aprendizaje Bayesiano

El razonamiento bayesiano ofrece un enfoque probabilístico para la inferencia. Se basa en la suposición de que las cantidades de interés están regidas por distribuciones de probabilidad y que se pueden tomar decisiones óptimas razonando sobre estas probabilidades junto con los datos observados. Este enfoque es crucial para el aprendizaje automático porque proporciona un método cuantitativo para evaluar la evidencia que respalda diferentes hipótesis. El razonamiento bayesiano no solo sirve como base para algoritmos de aprendizaje que manipulan directamente las probabilidades, sino que también ofrece un marco para analizar el funcionamiento de otros algoritmos que no manipulan explícitamente las probabilidades.

## Introduccion

El aprendizaje bayesiano es util por dos razones.

- Los algoritmos bayesianos de aprenziaje pueden calcular probabilidades de hipotesis, dado un conjunto de datos de entrenamiento. Por ejemplo el naive bayes es una tecnica practica para resolver diversos problemas. No solo eso, sino que los clasificadores bayesianos son competitvos con otras tecnicas de clasificacion, superadoras en algunos casos.

- Otra razon es que los modelos bayesianos son importantes dado que proveen una perspectiva util para el analisis de otros metodos que no manipulan probabilidades.

"La ausencia de evidencia no es evidencia de ausencia", esto quiere decir que algo no se manifieste en los datos no significa que no exista. Por ejemplo, si se busca un objeto en un lugar y no se lo encuentra, no significa que no este ahi, sino que no se lo encontro.

La inferencia estadistica es el proceso de utilizar el analisis de datos para deducir las propiedasd de una distribucion.

El enfoque bayesiano nos plantea que nada tiene probabilidad 0 o 1, sino que todo tiene una probabilidad entre 0 y 1.

## Repaso de probabilidad

Sucesos independientes: dos sucesos son independientes si la ocurrencia de uno no afecta la ocurrencia del otro.

P(A y B) = P(A) * P(B)
P(A | B) =  P(A y B) / P(B) = P(A)

Sucesos dependientes: dos sucesos son dependientes si la ocurrencia de uno afecta la ocurrencia del otro.

P(A | B) = P(A y B) / P(B) = P(A) * P(B | A) / P(B)

Formula de Bayes:

P(A y B) = P(B y A) => P(A | B) * P(B) = P(B | A) * P(A)

P(A | B) = P(B | A) * P(A) / P(B)

En los clasificadores:

h = hipotesis
D = datos

- P(h) es la probabilidad inicial de la ocurrencia de h.
- P(D | h) es la probabilidad de D, dado h.
- P(h | D) es la probabilidad posterior de h, dado D. (Probabilidad a posteriori)

P(h | D) = P(D | h) * P(h) / P(D)

Aplicando probiblidad total:

P(h | D) = P(D | h) * P(h) / P(D | h) * P(h) + P(D | no h) * P(no h)

La mejor hipotesis es la que tiene la mayor probabilidad a posteriori. Osea la que maximiza P(h | D). Si asumimos independencia entre los datos, entonces es una productoria de las probabilidades de cada dato dada la hipotesis. (Naive Bayes)

Se le conoce como MAP (Maximum a posteriori)
argmax P(h | D) = argmax P(D | h) * P(h)

Si se asume equiprobabilidad de las hipotesis, entonces la mejor hipotesis es la que maximiza P(D | h).

## Fortalezas.

Cada vez que se observa un dato nuevo, podemos incorporarlo a la hipotesis, para recalcular la probabilidad a posteriori y que sea mas precisa.

## Debilidades.

Requiere conocer valores iniciales de muchas probabilidades. Cuando estas no son conocidas, hay que estimarlas a partir de los datos.

## Acumulacion

La regla de Bayes es util para actualizar creeencias a medida que se obtienen nuevos datos. Esto se conoce como acumulacion de evidencia.

Hay que hacer algo entonces cuando no haya evidencia. Esto se conoce como suavizado.

## Suavizado de Probabilidades

- m-estimacion: Se le suma m a cada probabilidad. De esta forma evitamos que alguna probabilidad sea 0.

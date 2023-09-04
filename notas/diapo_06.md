# Aprendizaje basado en casos.

La idea es construir un clasificador perezoso, esto quiere decir que para clasificar un nuevo dato, se compara con todos los datos de entrenamiento que ya conozco.

Ventaja:
- para cada nueva instancia puedo obtener un nuevo modelo de clasificación.
- la descripcion de las instancias puede ser muy compleja.

Desventaja:
- el costo de clasificacion puede ser alto
- atributos irrelevantes pueden afectar la medida de similitud.

## K-NN (K-Nearest Neighbors)

Un ejemplo muy simple de clasificador perezoso es el K-NN.
Queremos aproximar un concepto utilizando las k instancias mas cercanas que deseamos clasificar.
Con la distancia euclidiana podemos calcular la distancia de una nueva instancia comparada con las demas.
Entonces el problema es encontrar ex que maximice la similitud con la nueva instancia.
En knn los ejemplos ams cercanos a la nueva instancia son lo mas importantes a considerar. Para eso se asigna un peso que modela la importancia de cada ejemplo.
el peso wi es inversamente proporcional a la distancia al cuadrado.
Otra idea es variar el k, a tomar un k mas grande se es mas tolerable a ruido y viceversa., generalmente se escoge un k impar para evitar empates.
A diferencia de los arboles de decision al aplicar la distancia, se tienen en cuenta todas las dimensiones que describen un atributo.
Si 2 atribhutos en 20 son relevantes, instancia que en realidad son muy diferentes pueden estar muy proximas en un espacio 20-dimensional.
Esto se conoce como la maldicion de la dimensionalidad.
Otra cosa, al usar la distancia euclidiana las magintudes de los valores nos juegan en contra..
Una posible solucion a esto es asignar pesos a los atributos menos relevantes. Como se hace esto?
Se puede penalizar atributos pseudo-uniformes o validacion cruzada. Cada clasificacion implica la consideracion de k elementos, por lo que el costo de clasificacion es alto.
El sesgo que tiene es que la clasificacion de una instancia es parecida a las dde sus k vecinos mas cercanos. Cercania -> similitud -> misma clase.
Esta bueno cuando hay que darle categorias a un categorias a un cjto de entrenamiento en crecimiento.
Ejemplo practico(sistema de recomendacion de peliculas)

## Regresion local ponderada.

Generalizacion de KNN, creando una aproximacion local de la funcion de clasificacion, construimos una funcion h que aproxime a los datos.
La funcion puede ser, lineal, cuadratica, etc.

Por ejemplo.

h([a1,..,an]) = w0 + w1*a1 + w2*a2 + ... + wn*an, buscamos el vector de pesos w que minimice el error cuadratico medio..

Para eso podemos considerar:

- los k vecinos mas cercanos
- todos los ejemplos de entrenamiento
- ponderar los k vecinos mas cercanos

Luego de clasificar una instancia, debemos hacer denuevo todo lo mismo para una siguiente conjunto.

## Razonamiento basado en casos.

Que sucede cuando no se puede aplicar formulas matematicas a las instancias de entrenamiento.
Por ejemplo dado un paquete de viaje, clasificar si el cliente esta satisfecho.

Este metodo no es numerico, es mas general, se generaliza con grafos.

## Algoritmos peresozos vs algoritmos ansiosos.

- Los algoritmos que vimos difieren el calculo de una hipotesis solo cuando hay una instancia nueva.
- Computan una aproximacion local de la funcion objetivo para resolver el problema de clasificacion de una nueva instancia.
- Los algoritmos ansiosos, pueden usar aproximacions sin embargo, quedan fijas al conjunto de entrenamineto

Dado un mismo espacio de hipotesis.

- los algoritmos perezosos tienen un mayor poder de adaptacion.
- el costo de clasificacion es mayor en los algoritmos perezosos.
- se necesitan estructuras efficientes para determinar los ejemplos cercanos.

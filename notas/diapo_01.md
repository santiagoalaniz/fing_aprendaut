# Aprendizaje Automatico

- Profesor Responsable: Diego Garat, dgarat@fing.edu.uy
- Email del curso: maa@fing.edu.uy

## Objetivos

Dar un panorama general de todas las areas relevantes del aprendizaje automatico (Machine Learning), Algoritmos Basicos no vamos a profundizar en ningun tema pero si aprender los conceptos basicos de cada uno de ellos.

Libro de referencia: Machine Learning, Tom Mitchell, 1997.
Aunque algunos temas escapan a este libro, sera indicado en cada caso.

## Temas

- Introducción
- Aprendizaje conceptual
- Árboles de decisión
- Metodología
- Aprendizaje bayesiano
- Aprendizaje basado en casos
- Aprendizaje no supervisado
- Aprendizaje por refuerzos
- Regresión lineal y logística
- Redes neuronales y Aprendizaje profundo

## Laboratorios

Son cuatro de caracter eliminatorio, a lo largo del semestre se va a otorgar cuatro dias de prorroga para entrega de laboratorios, no se aceptan laboratorios fuera de fecha de entrega.

## Prueba Final.

Miercoles 15 de Noviembre, 19:00hs.

# Introduccion.

## Que es el Aprendizaje Automatico?

Es un area de la computacion que incluye valores varias areas de la inteligencia artificial, estadistica, matematica, etc. El objetivo es desarrollar tecnicas que permitan a las computadoras "aprender" o "predecir" un comportamineto.

Se aprende a partir de ejemplos, instancias de datos que representan atributos a reconocer o predecir. Cada instancia puede venir ya estructurada, por ejemplo, una imagen, un texto, un audio, etc.

### Aprendizaje Supervisado

Dado un conjunto de datos de entrenamiento, cada uno con una etiqueta o clase, con un algoritmo se entrena un modelo que permita predecir la etiqueta de una nueva instancia.

Ejemplo visto: Clasificacion en base a una regresion lineal.
De este ejemplo aprendimos sobre ruido, sobreajuste, experimentacion, etc.

### Aprendizaje No Supervisado

Dado un conjunto de datos de entrenamiento, sin etiquetas, con un algoritmo se entrena un modelo que permita agrupar o clasificar las instancias.

- Como se cuantos grupos hay? Probando.

### Aprendizaje por Refuerzo

Dado un comportamiento a aprender (terminar un juego, ganar una partida, etc), se entrena un modelo que permita tomar decisiones en base a un estado del juego.

## Conclusion.

El aprendizaje automatico (segun el curso) es un programa que mejora su desempeño en una tarea a partir de la experiencia.

- Mejorar una tarea T (clasificar, predecir, etc)
- Respecto a una medida de desempeño P (precision, recall, etc)
- Basandono en la experiencia E (datos)

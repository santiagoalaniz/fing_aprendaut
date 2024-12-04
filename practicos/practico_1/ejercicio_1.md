# Ejercicio 1

## Parte (a)

Ejemplos donde las tecnicas de Aprendizaje Automatico pueden ser utiles:

- Reconocimiento de voz: Se puede entrenar un modelo para que reconozca la voz de una persona y asi poder identificarla. Esto se debe a que el registro vocal de una persona si bien no es unico, hay ciertas caracteristicas distintivas que permiten identificarla. Por ejemplo, la frecuencia de la voz, el timbre, el tono, etc.

Justificacion: El registro vocal de una persona es un dato que se puede obtener facilmente. Ademas, es un dato que no se puede falsificar facilmente.

- Mejora de una cadena logistica: Se puede entrenar un modelo para que prediga la demanda de un producto en un determinado periodo de tiempo. Esto permite a la empresa planificar mejor la produccion y el transporte de los productos.

Justificacion: La demanda de un producto en un determinado periodo de tiempo junto a otras variables pueden determinar tendencias ocultas que permitan predecir la demanda futura.

Ejemplos donde las tecnicas de Aprendizaje Automatico no son utiles:

- Prediccion de eventos estocasticos: Se puede entrenar un modelo para que prediga el resultado de un evento estocastico. Por ejemplo, predecir el resultado de un lanzamiento de una moneda.

Justificacion: Los eventos estocasticos son impredecibles por definicion. Por lo tanto, no se puede entrenar un modelo para que prediga el resultado de un evento estocastico.

- Diagnostico basado en sintomas generales: Se puede entrenar un modelo para que diagnostique una enfermedad basado en sintomas generales. Por ejemplo, predecir si una persona tiene fiebre, dolor de cabeza, etc.

Justificacion: Los sintomas generales pueden ser causados por muchas enfermedades distintas. Ademas, cada persona manifasta y percibe los sintomas de manera distinta, por ejemplo alta tolerancia al dolor ademas estos sintomas pueden ser causados por diversas enfermedades. Por lo tanto, no se puede entrenar un modelo para que diagnostique una enfermedad basado en sintomas generales.

## Parte (b)

Dado un correo electrónico entrante con ciertas características (por ejemplo, palabras clave, dirección del remitente, enlaces incluidos, etc.), el sistema debe clasificar este correo como "spam" o "no spam".

### Medida de performance:
Una métrica común para evaluar la eficacia de un filtro de spam es la precisión, que es el porcentaje de correos electrónicos clasificados correctamente (tanto spam como no spam) sobre el total de correos electrónicos clasificados. Otra métrica importante es el "recall" o sensibilidad, que mide el porcentaje de correos spam reales que fueron correctamente identificados por el sistema.

### Descripción de la función objetivo:
La función objetivo podría definirse como la maximización de la precisión y el recall del sistema.
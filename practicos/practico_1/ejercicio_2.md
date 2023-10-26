# Ejercicio 2

## Parte i

Hipotesis vista en el curso:

- *h* : \<Cielo, Temp, Humedad, Viento, Tmp. Agua, Tiempo>
- *c* : \<Juega>

*h* es la conjuncion de los atributos de la tabla de datos, y *c* es la columna de la tabla de datos que indica la funcion objetivo. El espacio de hipotesis H es el conjunto de todas las posibles conjunciones de los atributos de la tabla de datos. Entonces, el tamaño de $|H|$ es:

$ 
|H| = |Cielo| * |Temp| * |Humedad| * |Viento| * |Tmp. Agua| * |Tiempo| * |Juega|
$
<br>
$
|H| = 3 * 2 * 2 * 2 * 2 * 2 = 3 * 2^5 = 3 * 32 = 96
$

## Parte II

Para calcular el espacio de versiones hay que tener en cuenta que el mismo se puede definir mediante una cota inferior y superior de hipotesis mas especificas y generales.

```
S = [Soleado, Templado, ?, Fuerte, ?, ?]
G = [Soleado, ?, ?, ?, ?, ?], 
    [?, Templado, ?, ?, ?, ?], 
    [?, ?, Normal, ?, ?, ?], 
    [?, ?, ?, Suave, ?, ?], 
    [?, ?, ?, ?, Fría, ?], 
    [?, ?, ?, ?, ?, Cambiante]
```

## Parte III

Para determinar la respuesta a las instancias dadas, debemos verificar si cada instancia es consistente con las hipótesis específicas y generales que hemos obtenido después de aplicar el algoritmo de eliminación de candidatos.

**Instancia 5**: Soleado, Templado, Normal, Fuerte, Fría, Cambiante
- Es consistente con S.
- Es consistente con todas las hipótesis en G.
**Respuesta**: Sí

**Instancia 6**: Lluvioso, Frío, Normal, Suave, Templada, Sin cambios
- No es consistente con S.
- No es consistente con ninguna hipótesis en G.
**Respuesta**: No

**Instancia 7**: Soleado, Templado, Normal, Suave, Templada, Sin cambios
- No es consistente con S debido al viento.
- Es consistente con las hipótesis generales 1, 2, y 3 en G.
**Respuesta**: No se puede determinar con certeza. Podría ser "Sí" o "No".

**Instancia 8**: Soleado, Frío, Normal, Fuerte, Templada, Sin cambios
- No es consistente con S debido a la temperatura.
- Es consistente con las hipótesis generales 1 y 3 en G.
**Respuesta**: No se puede determinar con certeza. Podría ser "Sí" o "No".


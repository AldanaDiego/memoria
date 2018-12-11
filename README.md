# memoria
En este repositorio se encuentran las versiones finales de los experimentos realizados durante mi memoria.

## Visualización
Este código permite aplicar el algoritmo t-SNE sobre los datos y obtener una gráfica en dos dimensiones de ellos. Se crean gráficos donde las instancias se marcan según el valor discreto de las variables R y L, y también gráficos marcados según el valor continuo de dichas variables.

## Clasificación
Los benchmark de clasificación son una comparación de varios algoritmos para predecir aproximadamente las variables R y L, junto a varios métodos de reducción de dimensiones para preprocesar los datos. Realizando validación cruzada con 10 subconjuntos, se calcula la accuracy promedio para cada solución posible y se crean gráficos mostrando las accuracies para cada iteración de la validacion cruzada.

## Regresión
Similar al caso de clasificación, el benchmark de regresión compara varios algoritmos para predecir con exactitud las variables R y L. En este caso se utilizó reducción de dimensionalidad por eliminación de correlaciones y no se probaron otros métodos. Usando validación cruzada con 10 subconjuntos, se calculan el MAE y MSE promedio para cada algoritmo de regresión, y se crean gráficos con las métricas para cada iteración.

## Optimización de SVR
El último experimento consistió en una búsqueda de parámetros óptimos para la regresión de las variables mediante SVR. Se prueban los kernel lineal y radial, junto a varios valores de costo y gamma, y se selecciona la combinación que obtiene el mejor rendimiento.

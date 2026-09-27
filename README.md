# Clasificación de la cubierta forestal con árboles de decisión y SVM

Actividad 2 de la asignatura **Técnicas de Aprendizaje Automático** del Máster Universitario en Inteligencia Artificial (UNIR). El cuaderno resuelve un problema de clasificación multiclase sobre el dataset **Forest CoverType** y compara dos familias de modelos: árboles de decisión (`DecisionTreeClassifier`) y máquinas de vectores de soporte con kernel radial (`SVC`).

La pregunta es si se puede predecir qué especie domina una celda de bosque de 30 × 30 m usando solo variables cartográficas, sin imágenes de satélite. Se puede, y con bastante precisión, pero el resultado de la comparación depende de algo que suele pasarse por alto: cuántos datos puede usar cada algoritmo.

## Contenido del repositorio

```
.
├── Gabriel_Morejón_-_Actividad_2_CoverType_Solución.ipynb   # Cuaderno ejecutado: código, gráficos y respuestas
├── requirements.txt
├── README.md
└── covertype/                     # No versionado (ver .gitignore)
    ├── covtype.data.gz            # Fichero original de UCI
    └── covtype.info
```

## Dataset

| Característica        | Valor                                                                           |
| --------------------- | ------------------------------------------------------------------------------- |
| Fuente                | UCI Machine Learning Repository, id. 31                                         |
| Instancias            | 581.012                                                                         |
| Variables predictoras | 54 (10 continuas, 4 columnas binarias de `Wilderness_Area` y 40 de `Soil_Type`) |
| Variable objetivo     | `Cover_Type`, 7 clases                                                          |
| Valores nulos         | Ninguno                                                                         |
| Licencia              | CC BY 4.0                                                                       |

Clases: 1 Spruce/Fir, 2 Lodgepole Pine, 3 Ponderosa Pine, 4 Cottonwood/Willow, 5 Aspen, 6 Douglas-fir y 7 Krummholz. La distribución está muy desbalanceada: las clases 1 y 2 suman el 85,2 % de las observaciones y la clase 4 apenas el 0,47 %, una razón de 103 a 1.

### Carga de los datos

El cuaderno lee `covertype/covtype.data.gz` si existe y, si no, descarga los datos con `ucimlrepo`. En ambos casos comprueba que cada fila tenga exactamente un `1` en cada grupo one-hot.

> **Aviso sobre `ucimlrepo`.** `fetch_ucirepo(id=31)` no devuelve las columnas en el orden del fichero original. Entrega `Wilderness_Area1`, luego `Soil_Type1` a `Soil_Type40` y al final `Wilderness_Area2` a `Wilderness_Area4`. Si se renombran las columnas por posición, 43 de las 44 variables binarias quedan con la etiqueta equivocada: `Wilderness_Area4` aparece como `Soil_Type40`, `Soil_Type10` como `Soil_Type7`, y así sucesivamente. Los modelos no cambian, pero cualquier interpretación de las variables queda mal. El cuaderno selecciona las columnas por nombre para evitarlo.

## Instalación y ejecución

Probado con Python 3.12 y 3.13.

```bash
git clone <url-del-repositorio>
cd <carpeta-del-repositorio>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Para trabajar sin conexión, descarga `covtype.data.gz` desde [la página del dataset](https://archive.ics.uci.edu/dataset/31/covertype) y colócalo en `covertype/`. Después ejecuta **Kernel → Restart Kernel and Run All Cells**. En un equipo de un solo núcleo la ejecución completa tarda unos 2,5 minutos.

## Metodología

**Análisis descriptivo.** Tipos de variables y verificación de la codificación one-hot, distribución de clases, valores nulos, correlación de Pearson con la clase y razón de correlación η (más adecuada, porque la clase es nominal), matriz de correlaciones entre variables continuas y búsqueda de multicolinealidad. Las variables más influyentes se identifican contrastando un ANOVA F-test con la importancia de variables de un Random Forest de referencia.

**Árboles de decisión.** Partición estratificada 80/20 sobre el dataset completo (`random_state=1`). Cinco árboles con `criterion="gini"`, `min_samples_split=10`, `min_samples_leaf=2`, `max_features=None` y `max_depth` de 5, 10, 20, 50 y 100. Sin normalización: el cuaderno comprueba empíricamente que un árbol entrenado con datos escalados produce las mismas predicciones.

**SVM.** Se usan los primeros 11.340 registros del dataset. Según `covtype.info`, forman el subconjunto de entrenamiento del estudio original de Blackard y Dean y contienen exactamente 1.620 ejemplos de cada clase. Partición estratificada 80/20, `StandardScaler` ajustado solo con entrenamiento, kernel RBF, `gamma="scale"` y `C` ∈ {0,1; 1; 10; 100; 1000}. `C` se elige con validación cruzada estratificada de 5 particiones, con el escalado dentro de cada fold.

**Comparación en igualdad de condiciones.** Además de la comparación pedida, el cuaderno entrena un árbol con los mismos ejemplos balanceados que el SVM, eligiendo la profundidad con la misma validación cruzada, y evalúa ambos sobre el mismo conjunto de test.

## Resultados

### Árboles de decisión (dataset completo; test de 116.203 ejemplos con la distribución original)

| `max_depth` | Accuracy train | Accuracy test | F1 macro test | Profundidad real | Hojas  |
| ----------- | -------------- | ------------- | ------------- | ---------------- | ------ |
| 5           | 0,7023         | 0,7021        | 0,4820        | 5                | 32     |
| 10          | 0,7809         | 0,7775        | 0,6470        | 10               | 677    |
| 20          | 0,9335         | 0,9060        | 0,8516        | 20               | 10.014 |
| **50**      | 0,9790         | **0,9342**    | **0,8957**    | 41               | 17.324 |
| 100         | 0,9790         | 0,9342        | 0,8957        | 41               | 17.324 |

El árbol deja de crecer a los 41 niveles, así que `max_depth=50` y `max_depth=100` producen el mismo modelo. El ganador clasifica bien 108.553 de 116.203 ejemplos. Las clases minoritarias Aspen (F1 0,812) y Cottonwood/Willow (F1 0,839) son las más difíciles.

### SVM con kernel RBF (muestra balanceada; test de 2.268 ejemplos)

| `C`      | Accuracy train | Accuracy test | Accuracy CV (5 folds) | Vectores de soporte |
| -------- | -------------- | ------------- | --------------------- | ------------------- |
| 0,1      | 0,6403         | 0,6151        | 0,6232 ± 0,0075       | 7.006               |
| 1        | 0,7363         | 0,7063        | 0,7140 ± 0,0071       | 6.028               |
| 10       | 0,8265         | 0,7822        | 0,7804 ± 0,0117       | 5.206               |
| 100      | 0,8868         | 0,8051        | 0,8080 ± 0,0062       | 4.528               |
| **1000** | 0,9410         | **0,8086**    | **0,8116 ± 0,0044**   | 4.077               |

`C=100` y `C=1000` son prácticamente equivalentes: la diferencia en validación cruzada es menor que una desviación estándar, y `C=1000` sobreajusta más. Los errores se concentran en dos pares de especies que comparten hábitat: Spruce/Fir ↔ Lodgepole Pine y Ponderosa Pine ↔ Douglas-fir.

### Comparación

| Modelo                      | Ejemplos de entrenamiento | Test                   | Accuracy | F1 macro |
| --------------------------- | ------------------------- | ---------------------- | -------- | -------- |
| Árbol, dataset completo     | 464.809                   | 116.203, desbalanceado | 0,9342   | 0,8957   |
| Árbol, muestra balanceada   | 9.072                     | 2.268, balanceado      | 0,7637   | 0,7631   |
| SVM RBF, muestra balanceada | 9.072                     | 2.268, balanceado      | 0,8086   | 0,8068   |

Comparados sin más, el árbol parece muy superior. Con los mismos datos, el SVM gana en accuracy y en F1 macro, y en seis de las siete clases. La recomendación práctica sigue siendo el árbol, porque el coste de `SVC` crece al menos de forma cuadrática con el número de ejemplos y no puede aprovechar el dataset completo. En este problema, poder usar 51 veces más datos pesa más que la ventaja del algoritmo.

## Investigación

El cuaderno analiza un artículo posterior a 2015, publicado en revista y sin _deep learning_, que compara SVM (kernel RBF), Random Forest y k-NN en la clasificación de cobertura del suelo con distintos tamaños de muestra y grados de balanceo:

Thanh Noi, P., & Kappas, M. (2018). Comparison of Random Forest, k-Nearest Neighbor, and Support Vector Machine classifiers for land cover classification using Sentinel-2 imagery. _Sensors, 18_(1), 18. https://doi.org/10.3390/s18010018

## Uso de inteligencia artificial generativa

Se utilizó un asistente de IA (Claude, de Anthropic) para organizar explicaciones a partir de la documentación de scikit-learn, depurar código y, en una segunda fase, revisar el cuaderno frente a la rúbrica y aplicar las correcciones derivadas de esa revisión. La primera celda del cuaderno detalla el alcance de esa ayuda.

## Autor

**Gabriel Eduardo Morejón López**
Máster Universitario en Inteligencia Artificial, Universidad Internacional de La Rioja (UNIR)

## Referencias

- Blackard, J. (1998). _Covertype_ [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50K5N
- Blackard, J. A., & Dean, D. J. (1999). Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types from cartographic variables. _Computers and Electronics in Agriculture, 24_(3), 131-151.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. _Journal of Machine Learning Research, 12_, 2825-2830.
- Thanh Noi, P., & Kappas, M. (2018). Comparison of Random Forest, k-Nearest Neighbor, and Support Vector Machine classifiers for land cover classification using Sentinel-2 imagery. _Sensors, 18_(1), 18.

## Licencia

Código bajo licencia MIT. El dataset CoverType se distribuye bajo CC BY 4.0 y mantiene la autoría de Jock A. Blackard y Colorado State University.

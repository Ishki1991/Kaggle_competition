# W7-Kaggle_competition

![portada](https://github.com/Ironhack-Data-Madrid-Enero-2021/W7-Kaggle_competition/blob/main/images/PORTADA.jpg)

## Description

En este script utilizaermos el preprocesamiento, modelado y visualización de datos el cual requiere tener instaladas las siguientes librerías: numpy, pandas, scipy, scikit-learn, imblearn, matplotlib, y seaborn.

## Instructions


Se importa primero las librerías necesarias para el preprocesamiento de datos como imputación de datos (usando SimpleImputer, IterativeImputer, y KNNImputer), la estandarización de datos (usando MinMaxScaler, StandardScaler, y RobustScaler), y la codificación (usando LabelEncoder, OneHotEncoder, y OrdinalEncoder).

## Visualización

Aqui importamos matplotlib y seaborn para visualizar de datos. El script crea gráfico subplot para la detección de outliers y para medir el orden de las variables categoricas.


## Modelado

El script importa la librería RandomForestRegressor. El script guarda el modelo entrenado usando pickle.

## Conjunto de datos

Importa los datos usando pandas. Importa sample_submission.csv, test.csv, y train.csv para luego crea una copia del conjunto de datos. Realizamos un análisis exploratorio de los datos para verificar los tipos de datos, el número de valores nulos y el número duplicados.

Por último, con el método RandomForestRegressor compruebo las predicciones.

Nota: este README asume que el conjunto de datos se encuentra en una carpeta llamada "data" en el mismo directorio que el script de Python.

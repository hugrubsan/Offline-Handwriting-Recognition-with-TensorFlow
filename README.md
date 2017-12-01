# Offline Handwriting Recognition with Deep Learning implemented in TensorFlow.

Sistema de Deep Learning para el Reconocimiento de Palabras Manuscritas implementado en TensorFlow y entrenado con IAM Handwriting Database.

Sobre este sistema se realiza una validación cruzada y el test IAM.

## Estructura

### Ficheros Python:

- *clean_IAM.py*: Script para limpieza y el preprocesamiento de las imágenes.
- *ANN_model.py*: Modelo de la red neuronal implementada en TensorFlow.
- *cross-validation.py*: Script para la validación cruzada.
- *train.py*: Script para entrenar el modelo y almacenar los parámetros que consiguen un mejor resultado.
- *test.py*: Script para testear un modelo previamente entrenado.
- *hw_utils.py*: Funciones útiles en distintas partes del proyecto.

### CSV:

- *CSV/Cross-Validation*: Ficheros que contienen los nombres y las transcripciones de las imágenes de los distintos subconjuntos de entrenamiento y validación.
- *CSV/IAM test*: Ficheros que contienen los nombres y las transcripciones de las imágenes para el test IAM.
- *Data/appropriate_images.csv*: Fichero csv que contiene el nombre de las imágenes aptas(87.108).

### Fichero de configuración

Fichero que contiene todos los parámetros del proyecto:

```
"general": Parámetros generales del proyecto.
{
"raw_data_path": Ruta a las imágenes sin preprocesar.
"processed_data_path": Ruta para las imágenes preprocesadas.
"csv_path": Ruta al CSV de imágenes aptas.
"height": Altura de las imágenes preprocesadas.
"width": Anchura de las imágenes preprocesadas.
"dictionary": Diccionario para parsear las etiquetas.
}


"cnn-rnn-ctc": Hiperparámetros del modelo.
{
"kernel_size": Altura y anchura de los filtros de la CNN, filtros cuadrados.
"num_conv1" :Número de neuronas de la 1ª capa CNN.
"num_conv2" : Número de neuronas de la 2ª  capa CNN.
"num_conv3" : Número de neuronas de la 3ª capa CNN.
"num_conv4" : Número de neuronas de la 4ª capa CNN.
"num_conv5" : Número de neuronas de la 5ª capa CNN.
"num_rnn" : Número de neuronas de las capas RNNs.
"num_fc" : Número de neuronas de la 1ª capa Fullconnect.
"num_classes": Número de etiquetas, incluida la etiqueta "blanco".
"ctc_input_len": Longitud de la secuencia de entrada a la CTC.
}


"cross-validation": Parámetros para la validación cruzada.
{
"csv_path": Ruta a los CSVs para la validación.
"results_path": Ruta para los resultados.
"num_epochs": Número de épocas.
"validation_period": Periodo de épocas para realizar la validación del modelo.
"print_period": Periodo de épocas para la impresión por pantalla.
"batch_size": Tamaño del lote de muestras.
}


"IAM-test": Parámetros para el test IAM.
{
"csv_path": Ruta a los CSVs para el test.
"results_path": Ruta para los resultados.
"checkpoints_path": Ruta para almacenar el modelo entrenado.
"num_epochs": Número de épocas.
"validation_period": Periodo de épocas para realizar la validación del modelo.
"print_period":Periodo de épocas para la impresión por pantalla.
"batch_size" : Tamaño del lote de muestras.
}
```

## Primeros pasos.

### Requisitos Software
Python 3.6 y librerías:
- TensorFlow 1.3
- PIL
- Pandas
- Numpy
- Json
- Ast


### Instalación y preprocesado de datos.

Tras descargar o clonar el repositorio es necesario descargar el dataset de la [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) y descomprimirlo en el directorio "Offline-Handwriting-Recognition-with-TensorFlow\Data".

Una vez hemos conseguido el dataset ejecutamos:

```
python3 clean_IAM.py [path_config_file]
```
Si no se añade ninguna ruta al archivo de configuración se tomará la ruta por defecto "./config.json"

Este script selecciona las imágenes aptas para las pruebas, las reescala y le añade relleno hasta igualar sus dimensiones.

## Ejecución.

### Cross-validation.

Para realizar la validación cruzada del modelo solo es necesario ejecutar:

```
python3 cross-validation.py [path_config_file]
```

Este script realiza 10 validaciones con distintas subdivisiones del dataset original y almacena los resultados en formato CSV.


### Test IAM

El primer paso es entrenar el modelo con el dataset ofrecido por IAM con unas subdivisiones específicas. Para ello ejecutamos:

```
python3 train.py [path_config_file]
```

Este script realiza un entrenamiento del modelo y almacena los parámetros que mejor resultado han dado para el dataset de validación.

Una vez tenemos el modelo entrenado, obtenemos el resultado del test ejecutando:

```
python3 test.py [path_config_file]
```

El resultado se muestra por pantalla y las salidas del sistema se almacenan en CSV.

import os
import sys
from PIL import Image
import pandas as pd
import numpy as np


def scale_invert(raw_path, proc_path,height,width):

    """
    Función que escala e invierte cada imagen para almacenarlas en un directorio común.
    Se conservan las proporciones de la imagen original y se añade un relleno hasta alcanzar
    el ancho objetivo.

    Argumentos:

      - raw_path: Ruta de la imagen original. (String)
      - proc_path: Ruta donde almacenar la imagen procesada. (String)
      - height: Altura de las imágenes. (Int)
      - width: Anchura de la imágenes. (Int)

    """
    # Cargamos la imagen
    im = Image.open(raw_path)

    # Reescalamos
    raw_width, raw_height = im.size
    new_width = int(round(raw_width * (height / raw_height)))
    im = im.resize((new_width, height), Image.NEAREST)
    im_map = list(im.getdata())
    im_map = np.array(im_map)
    im_map = im_map.reshape(height, new_width).astype(np.uint8)

    # Rellenamos e invertimos los valores.
    data = np.full((height, width - new_width + 1), 255)
    im_map = np.concatenate((im_map, data), axis=1)
    im_map = im_map[:, 0:width]
    im_map = (255 - im_map)
    im_map = im_map.astype(np.uint8)
    im = Image.fromarray(im_map)

    # Almacenamos todas las imágenes en directorio común
    im.save(str(proc_path), "png")
    print("Processed image saved: " + str(proc_path))


def extract_training_batch(ctc_input_len,batch_size,im_path,csv_path):

    """
        Función que extrae un lote de imágenes y sus transcripciones de manera aleatoria para entrenar la ANN.

        Argumentos:

          - ctc_input_len: Longitud de la secuencia de entrada a la capa CTC. (Int)
          - batch_size: Tamaño del lote. (Int)
          - im_path: Ruta al directorio donde se almacenan las imágenes. (String)
          - csv_path: Ruta al dataset de entrenamiento. (Int)

        Salida:

          - batchx: Tensor que contiene las imágenes como matrices de entrada a la ANN.
            (Array de Floats: [batch_size, height, width, 1])
          - sparse: SparseTensor que contiene las etiquetas como valores enteros positivos. (SparseTensor: indice,values,shape)
          - transcriptions: Array con las transcripciones correspondientes a las imágenes de "batchx". (Array de Strings: [batch_size])
          - seq_len: Array con la longitud de la secuencia de entrada a la capa CTC, "ctc_input_len". (Array de Ints: [batch_size])
    """

    # Extraemos aleatoriamente un DataFrame de tamaño "batch_size" del Dataset de entrenamiento.
    df = pd.read_csv(csv_path, sep=",",index_col="index")
    df_sample=df.sample(batch_size).reset_index()

    # Declaramos las variables para la salida.
    batchx = []
    transcriptions = []
    index = []
    values=[]
    seq_len=[]

    # Creamos el lote a partir del Dataframe de muestras aleatorias.
    for i in range(batch_size):
        im_apt = df_sample.loc[i, ['image']].as_matrix()
        df_y =df_sample.loc[i, ['transcription']].as_matrix()
        for fich in im_apt:

            # Extraemos la imagen y la mapeamos en una matriz normalizada.
            fich = str(fich)
            fich = fich.replace("['", "").replace("']", "")
            im = Image.open(im_path + fich + ".png")
            width, height = im.size
            im_map = list(im.getdata())
            im_map = np.array(im_map)
            im_map = im_map / 255
            result=im_map.reshape(height, width,1)
            batchx.append(result)

            # Extraemos las etiquetas parseando la transcripción.
            original=""
            for n in list(str(df_y)):
                if n == n.lower() and n == n.upper():
                    if n in "0123456789":
                        values.append(int(n))
                        original = original + n
                elif n == n.lower():
                    values.append(int(ord(n) - 61))
                    original = original + n
                elif n == n.upper():
                    values.append(int((ord(n) - 55)))
                    original = original + n

            # Añadimos el indice del SparseTensor.
            for j in range(len(str(df_y))-4):
                index.append([i,j])

            # Añadimos las transcripciones y la longitud de la secuencia de entrada a la CTC.
            transcriptions.append(original)
            seq_len.append(ctc_input_len)

    # Creamos el Array que contiene todas las imagenes normalizadas el lote, entrada de la ANN.
    batchx = np.stack(batchx, axis=0)

    # Creamos el SparseTensor con el indice, las etiquetas que representan cada caracter y la longitud máxima de palabra
    shape=[batch_size,18]
    sparse=index,values,shape

    return batchx, sparse, transcriptions, seq_len


def extract_ordered_batch(ctc_input_len,batch_size,im_path,csv_path,cont):

    """
        Función que extrae un lote de imágenes y sus transcripciones de manera ordenada para validar o testear la ANN.

        Argumentos:

          - ctc_input_len: Longitud de la secuencia de entrada a la capa CTC. (Int)
          - batch_size: Tamaño del lote. (Int)
          - im_path: Ruta al directorio donde se almacenan las imágenes. (String)
          - csv_path: Ruta al dataset de validación. (Int)
          - cont: Índice auxiliar que permite la extracción de lotes de manera ordenada. (Int)

        Salida:

          - batchx: Tensor que contiene las imágenes como matrices de entrada a la ANN.
            (Array de Floats: [batch_size, height, width, 1])
          - sparse: SparseTensor que contiene las etiquetas como valores enteros positivos. (SparseTensor: indice,values,shape)
          - transcriptions: Array con las transcripciones correspondientes a las imágenes de "batchx". (Array de Strings: [batch_size])
          - seq_len: Array con la longitud de la secuencia de entrada a la capa CTC, "ctc_input_len". (Array de Ints: [batch_size])
          - num_samples: Número de muestras extraidas. (Int)
    """

    # Extraemos secuencialmente un DataFrame de tamaño "batch_size" del Dataset.
    df = pd.read_csv(csv_path, sep=",",index_col="index")
    df_sample=df.loc[int(cont*batch_size):int((cont+1)*batch_size)-1,:].reset_index()
    num_samples=int(len(df_sample.axes[0]))



    # Declaramos las variables para la salida.
    batchx = []
    transcriptions = []
    index = []
    values=[]
    seq_len=[]

    # Creamos el lote a partir del Dataframe de muestras aleatorias.
    if len(df_sample.axes[0]) is not 0:
        for i in range(len(df_sample.axes[0])):
            im_apt = df_sample.loc[i, ['image']].as_matrix()
            df_y =df_sample.loc[i, ['transcription']].as_matrix()
            for fich in im_apt:

                # Extraemos la imagen y la mapeamos en una matriz normalizada.
                fich = str(fich)
                fich = fich.replace("['", "").replace("']", "")
                im = Image.open(im_path + fich + ".png")
                width, height = im.size
                im_map = list(im.getdata())
                im_map = np.array(im_map)
                im_map = im_map / 255
                result=im_map.reshape(height, width,1)
                batchx.append(result)

                # Extraemos las etiquetas parseando la transcripción.
                original=""
                for n in list(str(df_y)):
                    if n == n.lower() and n == n.upper():
                        if n in "0123456789":
                            values.append(int(n))
                            original=original+n
                    elif n==n.lower():
                        values.append(int(ord(n)-61))
                        original = original + n
                    elif n==n.upper():
                        values.append(int((ord(n)-55)))
                        original = original + n

                # Añadimos el indice del SparseTensor.
                for j in range(len(str(df_y))-4):
                    index.append([i,j])

                # Añadimos las transcripciones y la longitud de la secuencia de entrada a la CTC.
                transcriptions.append(original)
                seq_len.append(ctc_input_len)

        # Creamos el Array que contiene todas las imagenes normalizadas el lote, entrada de la ANN.
        batchx=np.stack(batchx, axis=0)

    # Creamos el SparseTensor con el indice, las etiquetas que representan cada caracter y la longitud máxima de palabra
    shape=[batch_size,18]
    sparse=index,values,shape

    return batchx, sparse, transcriptions, seq_len, num_samples




def validation(curr_epoch,ctc_input_len, batch_size, im_path, csv_path, inputs, targets, keep_prob, seq_len, session, cost, ler):

    """
        Función que realiza la validación de la ANN sobre un dataset concreto.

        Argumentos:

          - curr_epoch: Época actual. (Int)
          - ctc_input_len: Longitud de la secuencia de entrada a la capa CTC. (Int)
          - batch_size: Tamaño del lote. (Int)
          - im_path: Ruta al directorio donde se almacenan las imágenes. (String)
          - csv_path: Ruta al dataset de validación. (Int)
          - inputs: Placeholder de la entrada del model. (placeholder)
          - targets: Placeholder de las salidas objetivo. (placeholder)
          - keep_prob: Placeholder para la probabilidad de dropout. (placeholder)
          - seq_len: Placeholder para la longitud de la secuencia de entrada a la capa CTC. (placeholder)
          - session: Sesión actual de TensorFlow. (Session)
          - cost: Tensor para la salida del error de la CTC. (Tensor: [1])
          - ler: Tensor para la salida del LER. (Tensor:[1])

        Salida:

          - val_tuple: Resultado de la validación del modelo en una época completa. (Tuple: {'epoch','cost','LER'})
    """

    # Variables auxiliares.
    cont = 0
    total_val_cost = 0
    total_val_ler = 0

    # Bucle para realizar la validación sobre el Dataset completo
    while cont >= 0:
        # Extraemos los lotes de forma secuencial mediante "cont"
        val_inputs, val_targets, val_original, val_seq_len, num_samples = extract_ordered_batch(
            ctc_input_len, batch_size, im_path, csv_path, cont)

        # Si el número de muestras extraidas el igual a "batch_size" se ha extraido un lote completo.
        if num_samples == batch_size:
            val_feed = {inputs: val_inputs,
                    targets: val_targets,
                    keep_prob: 1,
                    seq_len: val_seq_len}
            val_cost, val_ler = session.run([cost, ler], val_feed)
            total_val_cost += val_cost
            total_val_ler += val_ler
            cont += 1

        # No se ha podido extraer ningún lote más y por lo tanto, se calcula la media de "cost" y "ler"
        elif num_samples == 0:
            val_tuple = {'epoch': [curr_epoch], 'val_cost': [total_val_cost / (cont + 1)],
                    'val_ler': [total_val_ler / (cont + 1)]}
            cont = -1

        # No se ha podido extraer el lote completo, no quedan suficientes muestras en el Dataset y por lo tanto,
        # se calcula la media de "cost" y "ler".
        else:
            val_feed = {inputs: val_inputs,
                    targets: val_targets,
                    keep_prob: 1,
                    seq_len: val_seq_len}
            val_cost, val_ler = session.run([cost, ler], val_feed)
            total_val_cost += val_cost
            total_val_ler += val_ler
            val_tuple = {'epoch': [curr_epoch], 'val_cost': [total_val_cost / (cont + 1)],
                    'val_ler': [total_val_ler / (cont + 1)]}
            cont = -1

    return val_tuple


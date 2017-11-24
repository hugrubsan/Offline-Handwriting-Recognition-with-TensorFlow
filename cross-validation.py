
import os
import sys
from random import randint
import tensorflow as tf
import hw_utils
import pandas as pd
import ANN_model
import json
import ast

def run_ctc():

    # Ruta del archivo de configuración, pasada por argumentos o por defecto "./config.json".
    if len(sys.argv) == 1:
        print("Execution without arguments, config file by default: ./config.json")
        config_file = str('./config.json')

    elif len(sys.argv) == 2:
        print("Execution with arguments, config file:" + str(sys.argv[1]))
        config_file = str(sys.argv[1])

    else:
        print()
        print("ERROR")
        print("Wrong number of arguments. Execute:")
        print(">> python3 cross-validation.py [path_config_file]")
        exit(1)

    # Cargamos el archivo de configuración
    try:
        config = json.load(open(config_file))
    except FileNotFoundError:
        print()
        print("ERROR")
        print("No such config file : " + config_file)
        exit(1)
    # Si el directorio destino no existe, se crea.
    if not os.path.exists(str(config["cross-validation"]["results_path"])):
        os.mkdir(str(config["cross-validation"]["results_path"]))


    # Extraemos las variables generales para la cross-validation
    im_path=str(config["general"]["processed_data_path"])
    csv_path=str(config["cross-validation"]["csv_path"])
    results_path=str(config["cross-validation"]["results_path"])
    batch_size = int(config["cross-validation"]["batch_size"])
    num_epochs = int(config["cross-validation"]["num_epochs"])
    val_period = int(config["cross-validation"]["validation_period"])
    print_period = int(config["cross-validation"]["print_period"])
    height = int(config["general"]["height"])
    width = int(config["general"]["width"])
    dct=ast.literal_eval(str(config["general"]["dictionary"]))

    # Extraemos los parametros del modelo a validar
    kernel_size=int(config["cnn-rnn-ctc"]["kernel_size"])
    num_conv1=int(config["cnn-rnn-ctc"]["num_conv1"])
    num_conv2=int(config["cnn-rnn-ctc"]["num_conv2"])
    num_conv3=int(config["cnn-rnn-ctc"]["num_conv3"])
    num_conv4=int(config["cnn-rnn-ctc"]["num_conv4"])
    num_conv5=int(config["cnn-rnn-ctc"]["num_conv5"])
    num_rnn=int(config["cnn-rnn-ctc"]["num_rnn"])
    num_fc=int(config["cnn-rnn-ctc"]["num_fc"])
    num_classes=int(config["cnn-rnn-ctc"]["num_classes"])
    ctc_input_len=int(config["cnn-rnn-ctc"]["ctc_input_len"])

    # Creamos el modelo ANN
    model = ANN_model.CNN_RNN_CTC(kernel_size, num_conv1, num_conv2, num_conv3, num_conv4,
                               num_conv5, num_rnn, num_fc, height, width, num_classes)
    graph=model[0]
    inputs=model[1]
    targets=model[2]
    keep_prob=model[3]
    seq_len=model[4]
    optimizer=model[5]
    cost=model[6]
    ler=model[7]
    decoded=model[8]


    # Bucle para realizar la validación sobre los 10 Datasets.
    for i in range(10):

        # Declaramos los DataFrames para almacenar el resultado de cada validación
        train_result = pd.DataFrame()
        val_result = pd.DataFrame()

        # Creamos la sesión con el modelo previamente cargado e inicializamos la variables.
        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()

            # Bucle de épocas.
            for curr_epoch in range(num_epochs):

                # Extraemos un lote aleatorio del Dataset de entrenamiento.
                train_inputs, train_targets, original, train_seq_len = hw_utils.extract_training_batch(ctc_input_len,batch_size,im_path,csv_path + "train" + str(i + 1) + ".csv")
                feed = {inputs: train_inputs, targets: train_targets, keep_prob: 0.5, seq_len: train_seq_len}

                # Ejecutamos "optimizer", minimizando el error para el lote extraido.
                _ = session.run([optimizer], feed)

                # Comprobamos el periodo de validación para el modelo.
                if curr_epoch % val_period == 0:
                    # Calculamos el error de la CTC y el LER para el dataset de entrenamiento y almacenamos los resultados.
                    train_cost, train_ler = session.run([cost, ler], feed)
                    train_tuple = {'epoch': [curr_epoch], 'train_cost': [train_cost], 'train_ler': [train_ler]}
                    train_result = pd.concat([train_result, pd.DataFrame(train_tuple)])

                    # Realizamos la validación del modelo y almacenamos los resultados.
                    val_tuple=hw_utils.validation(curr_epoch,ctc_input_len, batch_size, im_path, csv_path + "validation" + str(i + 1) + ".csv", inputs, targets, keep_prob, seq_len, session, cost, ler)
                    val_result = pd.concat([val_result, pd.DataFrame(val_tuple)])

                # Comprobamos el periodo de impresión de ejemplos.
                if curr_epoch % print_period == 0:

                    # Imprimimos la salida del modelo para 10 ejemplos al azar del dataset de validación.
                    print("Epoch: " + str(curr_epoch))
                    print("Examples:")
                    for j in range(10):

                        # Extraemos una muestra.
                        prob_inputs, prob_targets, prob_original, prob_seq_len, _ = hw_utils.extract_ordered_batch(ctc_input_len,1,im_path,csv_path + "validation" + str(i + 1) + ".csv",randint(0,8500))
                        prob_feed = {inputs: prob_inputs,
                                     targets: prob_targets,
                                     keep_prob: 1,
                                     seq_len: prob_seq_len}

                        # Obtenemos la salida y la mapeamos como una palabra para imprimirla por pantalla.
                        prob_d = session.run(decoded[0], feed_dict=prob_feed)
                        output = str(list(map(dct.get, list(prob_d.values))))
                        for ch in ["['", "']", "', '"]:
                            output = output.replace(ch, "")
                            prob_original=str(prob_original).replace(ch, "")
                        print("Target: " + prob_original + "       Model Output: " + output)

            # Cerramos para la siguiente validación.
            session.close()

        # Almacenamos los resultados
        val_result.to_csv(results_path + "validation_result" + str(i + 1) + ".csv",index=False)
        train_result.to_csv(results_path + "training_result" + str(i + 1) + ".csv",index=False)
        print("Available results number " + str(i + 1)+" in "+ results_path)

    print("THE CROSS-VALIDATION IS OVER")

if __name__ == '__main__':
    run_ctc()


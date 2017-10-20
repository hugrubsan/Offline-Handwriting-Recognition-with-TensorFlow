
import os
import sys
from random import randint
import tensorflow as tf
import hw_utils
import pandas as pd
import models
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

    # Carga del archivo de configuración
    try:
        data = json.load(open(config_file))
    except FileNotFoundError:
        print()
        print("ERROR")
        print("No such config file : " + config_file)
        exit(1)

    if not os.path.exists(str(config["cross-validation"]["results_path"])):
        os.mkdir(str(config["cross-validation"]["results_path"]))


    im_path=str(config["general"]["processed_data_path"])
    csv_path=str(config["cross-validation"]["csv_path"])
    results_path=str(config["cross-validation"]["results_path"])
    batch_size = int(config["cross-validation"]["batch_size"])
    num_epochs = int(config["cross-validation"]["num_epochs"])
    height = int(config["general"]["height"])
    width = int(config["general"]["width"])
    dct=ast.literal_eval(str(config["general"]["dictionary"]))


    kernel_size=int(config["cnn-brnn-ctc"]["kernel_size"])
    num_conv1=int(config["cnn-brnn-ctc"]["num_conv1"])
    num_conv2=int(config["cnn-brnn-ctc"]["num_conv2"])
    num_conv3=int(config["cnn-brnn-ctc"]["num_conv3"])
    num_conv4=int(config["cnn-brnn-ctc"]["num_conv4"])
    num_conv5=int(config["cnn-brnn-ctc"]["num_conv5"])
    num_rnn=int(config["cnn-brnn-ctc"]["num_rnn"])
    num_fc=int(config["cnn-brnn-ctc"]["num_fc"])
    num_classes=int(config["cnn-brnn-ctc"]["num_classes"])
    ctc_input_len=int(config["cnn-brnn-ctc"]["ctc_input_len"])


    model = models.CNN_BNN_CTC(kernel_size, num_conv1, num_conv2, num_conv3, num_conv4,
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


    for i in range(10):
        result_train = pd.DataFrame()
        result_validation = pd.DataFrame()

        with tf.Session(graph=graph) as session:

            tf.global_variables_initializer().run()

            for curr_epoch in range(num_epochs):

                train_inputs, train_targets, original, train_seq_len = hw_utils.extract_batch_train(ctc_input_len,batch_size,im_path,csv_path + "train" + str(i + 1) + ".csv")
                feed = {inputs: train_inputs, targets: train_targets, keep_prob: 0.5, seq_len: train_seq_len}
                _ = session.run([optimizer], feed)

                if curr_epoch % 50 == 0:
                    train_cost, train_ler = session.run([cost, ler], feed)
                    tuple = {'epoch': [curr_epoch], 'train_cost': [train_cost], 'train_ler': [train_ler]}
                    result_train = pd.concat([result_train, pd.DataFrame(tuple)])
                    tuple=hw_utils.validation(curr_epoch,ctc_input_len, batch_size, im_path, csv_path + "validation" + str(i + 1) + ".csv", inputs, targets, keep_prob, seq_len, session, cost, ler)
                    result_validation = pd.concat([result_validation, pd.DataFrame(tuple)])


                if curr_epoch % 1000 == 0:
                    print("Epoch: " + str(curr_epoch))
                    print("Examples:")
                    for j in range(10):
                        prob_inputs, prob_targets, prob_original, prob_seq_len, _ = hw_utils.extract_batch_validation(ctc_input_len,1,im_path,csv_path + "validation" + str(i + 1) + ".csv",randint(0,8500))
                        prob_feed = {inputs: prob_inputs,
                                     targets: prob_targets,
                                     keep_prob: 1,
                                     seq_len: prob_seq_len}
                        prob_d = session.run(decoded[0], feed_dict=prob_feed)
                        output = str(list(map(dct.get, list(prob_d.values))))
                        for ch in ["['", "']", "', '"]:
                            output = output.replace(ch, "")
                        print("Target: " + str(prob_original) + "       Model Output: " + output)
            session.close()
        result_validation.to_csv(results_path + "validation_result" + str(i + 1) + ".csv")
        result_train.to_csv(results_path + "train_result" + str(i + 1) + ".csv")
        print("AVAILABLE RESULTS " + str(i + 1))


if __name__ == '__main__':
    run_ctc()


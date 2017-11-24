#!/usr/bin/env python
# -*- coding: utf-8

import os
import sys
from PIL import Image
import pandas as pd
import numpy as np
import json
import hw_utils

def main():

    # Ruta del archivo de configuración, pasada por argumentos o por defecto "./config.json".
    if len(sys.argv) == 1:
        print("Execution without arguments, config file by default: ./config.json")
        config_file=str('./config.json')
    elif len(sys.argv) == 2:
        print("Execution with arguments, config file:" +str(sys.argv[1]))
        config_file = str(sys.argv[1])
    else:
        print()
        print("ERROR")
        print("Wrong number of arguments. Execute:")
        print(">> python3 clean_IAM.py [path_config_file]")
        exit(1)


    # Cargamos el archivo de configuración
    try:
        data = json.load(open(config_file))
    except FileNotFoundError:
        print()
        print("ERROR")
        print("No such config file : " + config_file)
        exit(1)


    # Si el directorio destino no existe, se crea.
    if not os.path.exists(str(data["general"]["processed_data_path"])):
        os.mkdir(str(data["general"]["processed_data_path"]))

    # Lista con todos los ficheros del directorio.
    lstDir = os.walk(str(data["general"]["raw_data_path"]))

    # Se leen del csv los nombres de las imagenes aptas.
    df = pd.read_csv(str(data["general"]["csv_path"]), sep=",",index_col="index")
    df = df.loc[:, ['image']]
    lstIm = df.as_matrix()

    # Recorremos el directorio donde estan almacenadas las imagenes de la IAM.
    for root, dirs, files in lstDir:
        for file in files:
            (name, ext) = os.path.splitext(file)
            # Comprobamos si el fichero está en el array de imagenes aptas.
            if name in lstIm:
                # Se ejecuta la función "scale_invert"
                hw_utils.scale_invert(str(root)+str("/")+str(name+ext),str(data["general"]["processed_data_path"])+str(name+ext),int(data["general"]["height"]),int(data["general"]["width"]))


if __name__ == "__main__":
    main()

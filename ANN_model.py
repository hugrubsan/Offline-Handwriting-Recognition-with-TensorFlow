import tensorflow as tf

def CNN_RNN_CTC(kernel_size, num_conv1, num_conv2, num_conv3, num_conv4, num_conv5, num_rnn, num_fc, HEIGHT, WIDTH, num_classes):
    """
        Función que construye el modelo ANNs basado en capas convolucionales, capas recurrentes y una última capa CTC.
        Arquitectura compuesta por 5 Capas CNN-MAXPOOL-DROP, dos RNNs que trabajan en paralelo
        recorriendo una misma dimensión pero en sentidos opuestos, una Fullconnect-CTC que nos devuelve el error cometido.

        Argumentos:

          - kernel_size: Tamaño del kernel para las CNN. (Int)
          - num_conv1: Número de neuronas de la 1ª CNN. (Int)
          - num_conv2: Número de neuronas de la 2ª CNN. (Int)
          - num_conv3: Número de neuronas de la 3ª CNN. (Int)
          - num_conv4: Número de neuronas de la 4ª CNN. (Int)
          - num_conv5: Número de neuronas de la 5ª CNN. (Int)
          - num_rnn: Número de neuronas de la RNN. (Int)
          - num_fc: Número de neuronas de la 1ª Fullconnect. (Int)
          - HEIGHT: Altura de las imagenes. (Int)
          - WIDTH: Anchura de las imágenes. (Int)
          - num_classes: Número de etiquetas, reales + "blaco". (Int)

        Salida:

          - graph: Grafo que contiene la arquitectura del modelo. (Graph)
          - inputs: Placeholder para la entrada. (Placeholder)
          - targets: Placeholder para las salidas objetivo. (Placeholder)
          - keep_prob: Placeholder para la probabilidad de dropout. (Placeholder)
          - seq_len: Placeholder para longitud de la secuencia de entrada a la CTC . (Placeholder)
          - optimizer: Operador para minimizar el error del modelo. (Operation)
          - cost: Operador para obtener el error del modelo. (Operation)
          - ler: Operador para obtener el LER del modelo. (Operation)
          - decoded: Operador para obtener la decodificación de la salida del modelo. (Operation)
    """


    graph = tf.Graph()
    with graph.as_default():

        # PLACEHOLDERS
        # Entrada: Tensor [tamaño de bacth, altura imagen, anchura imagen, canal] 
        inputs = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH,1])

        # Longitud de la secuencia de entrada a la CTC: Tensor [tamaño de bacth]
        seq_len = tf.placeholder(tf.int32, [None])

	    # Problabilidad de Dropout: float32
        keep_prob = tf.placeholder(tf.float32)

	    # Etiquetas objetivo: SparseTensor (indice, etiquetas, [tamaño de bacth, longitud máxima de palabra])
        targets = tf.sparse_placeholder(tf.int32)


        # VARIABLES
	    # Filtros y bias de CNNs
        w_conv1 = tf.Variable(tf.random_normal([kernel_size, kernel_size, 1,num_conv1], stddev=0.01))
        b_conv1 = tf.Variable(tf.random_normal([num_conv1], stddev=0.01))
        w_conv2 = tf.Variable(tf.random_normal([kernel_size, kernel_size, num_conv1, num_conv2], stddev=0.01))
        b_conv2 = tf.Variable(tf.random_normal([num_conv2], stddev=0.01))
        w_conv3 = tf.Variable(tf.random_normal([kernel_size, kernel_size, num_conv2,num_conv3], stddev=0.01))
        b_conv3 = tf.Variable(tf.random_normal([num_conv3], stddev=0.01))
        w_conv4 = tf.Variable(tf.random_normal([kernel_size, kernel_size, num_conv3, num_conv4], stddev=0.01))
        b_conv4 = tf.Variable(tf.random_normal([num_conv4], stddev=0.01))
        w_conv5 = tf.Variable(tf.random_normal([kernel_size, kernel_size, num_conv4, num_conv5], stddev=0.01))
        b_conv5 = tf.Variable(tf.random_normal([num_conv5], stddev=0.01))

	    # Pesos de RNN
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_rnn, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_rnn, forget_bias=1.0)

	    # Pesos de FULLCONNECT
        w1 = tf.Variable(tf.random_normal([2*num_rnn, num_fc], stddev=0.01))
        b1 = tf.Variable(tf.random_normal([num_fc], stddev=0.01))
        w2 = tf.Variable(tf.random_normal([num_fc,num_classes], stddev=0.01))
        b2 = tf.Variable(tf.random_normal([num_classes], stddev=0.01))



        # ARQUITECTURA
	    # Normalización de la entrada
        inputs = tf.nn.l2_normalize(inputs, [1, 2])

	    #CAPA 1 CNN-MAXPOOL-DROP
        h_conv1 = tf.nn.relu(tf.nn.conv2d(inputs, w_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        h_pool1=tf.nn.l2_normalize(h_pool1,[1,2])
        h_pool1=tf.nn.dropout(h_pool1,keep_prob=keep_prob)

        # CAPA 2 CNN-MAXPOOL-DROP
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        h_pool2=tf.nn.l2_normalize(h_pool2,[1,2])
        h_pool2=tf.nn.dropout(h_pool2,keep_prob=keep_prob)

        # CAPA 3 CNN-MAXPOOL-DROP
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, w_conv3, strides=[1,1,1,1], padding='SAME') + b_conv3)
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        h_pool3=tf.nn.l2_normalize(h_pool3,[1,2])
        h_pool3=tf.nn.dropout(h_pool3,keep_prob=keep_prob)

        # CAPA 4 CNN-MAXPOOL-DROP
        h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool3, w_conv4, strides=[1,1,1,1], padding='SAME') + b_conv4)
        h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        h_pool4=tf.nn.l2_normalize(h_pool4,[1,2])
        h_pool4=tf.nn.dropout(h_pool4,keep_prob=keep_prob)

        # CAPA 5 CNN-MAXPOOL-DROP
        h_conv5 = tf.nn.relu(tf.nn.conv2d(h_pool4, w_conv5, strides=[1,1,1,1], padding='SAME') + b_conv5)
        h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        h_pool5=tf.nn.l2_normalize(h_pool5,[1,2])
        h_pool5=tf.nn.dropout(h_pool5,keep_prob=keep_prob)

	    # CAPA 6 RNNs
        outputs=tf.transpose(h_pool5, (2,0,1,3))
        outputs=tf.reshape(outputs, (int(WIDTH/(2**5)),-1,int(HEIGHT*num_conv5/(2**5))))
        outputs=tf.transpose(outputs, (1,0,2))
        outputs, _= tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, outputs,dtype=tf.float32)
        outputs=tf.concat(outputs,2)
        outputs=tf.transpose(outputs, (1,0,2))        
        outputs=tf.reshape(outputs, (-1,2*num_rnn))

        # CAPA 7 FULLCONNECT-CTC_Loss
        logits = tf.matmul(outputs, w1) + b1
        logits =  tf.matmul(logits, w2) + b2
        logits = tf.reshape(logits, (int(WIDTH/(2**5)),-1,num_classes))
        loss = tf.nn.ctc_loss(targets, logits, seq_len,preprocess_collapse_repeated=True)
        cost = tf.reduce_mean(loss)



        # Optimización para minimizar el error
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


	    # Decodificador para extraer la secuencia de caracteres
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

        # Error: Label Error Rate
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    return graph, inputs, targets, keep_prob, seq_len, optimizer, cost, ler, decoded

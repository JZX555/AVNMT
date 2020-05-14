import tensorflow as tf
import tensorflow_datasets
from hyper_and_conf import conf_metrics
import numpy as np

# import tensorflow_addons as tfa

import core_data_SRCandTGT


# DATA_PATH = 'C:/Users/50568/Desktop/鎴�/maching cafe/unmt/corpus'

# PAD_ID_int64 = tf.cast(0, tf.int64)

# dm = core_data_SRCandTGT.DatasetManager([DATA_PATH + "/europarl-v7.fr-en.fr"],
#                                 [DATA_PATH + "/europarl-v7.fr-en.en"],
#                                 batch_size=16,
#                                 shuffle=100)

# test = dm.get_raw_train_dataset(shuffle=False) 
# # test = test.padded_batch(2, padded_shapes = ([30], [30]), padding_values = (PAD_ID_int64, PAD_ID_int64))

# for (batch, (x, y)) in enumerate(test):
#     print(x)
#     print(dm.decode(x))
#     print(y)
#     print(dm.decode(y))
#     print(dm.decode([9151, 2863 ,   5 , 389   , 5 ,  29  , 38, 3092  , 64   ,54, 4119   , 7  ,  2  ,  0,
#     0 ,0,0,0,0,0]))
#     l = [9151, 2863, 5, 389, 5, 29, 38, 3092, 64]
#     print(dm.decode(l))  
#     print(dm.decode([9151, 2863 ,   5 , 389   , 5 ,  29  , 38, 3092  , 64, 0 , 0]))    
#     print(dm.decode([9151, 2863 ,   5 , 389   ])) 
#     break

# a = tf.convert_to_tensor([[2, 2, 3, 2, 2, 1, 4, 4], [2, 2, 3, 2, 2, 1, 4, 4]])
# b = tf.convert_to_tensor([[2, 3, 3, 1, 0, 0],[2, 2, 3, 1, 0, 0]])
# d = tf.convert_to_tensor([[2, 3, 3, 1, 0, 0],[2, 2, 3, 1, 0, 0]])

# bleu, c = conf_metrics.approx_bleu(a, b)
# accuracy, _ = conf_metrics.padded_accuracy(a, b)
# print('bleu is: {}'.format(bleu))
# print('accuracy is: {}'.format(tf.reduce_mean(accuracy)))
# print('c is: {}'.format(c))

# cnt = 0
# id = 0

# with open('C:/Users/50568/Desktop/鎴�/data/training-parallel-commoncrawl', 'r', encoding='UTF-8') as f:
#     print('begin')
#     while True:
#         line = f.readline()
#         if not line:
#             break

#         if cnt == 0:
#             sub_f = tf.io.gfile.GFile('C:/Users/50568/Desktop/鎴�/data/commoncrawl/sub_' + str(id), 'w')

        
#         if cnt <= 3000000:
#             sub_f.write(line)
#             cnt += 1

#         else:
#             print('generate sub datasets{}'.format(id))
#             cnt = 0
#             id += 1
#             sub_f.close()


# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# print(tf.__version__)
# print(np.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
# labels = tf.cast([[0, 1, 2], [0, 1, 2]], tf.int32)
# logits = tf.cast([[[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]], [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]]], tf.float32)
# print(logits)

# #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))
# loss = cce(labels, logits)
# print(loss)

# a = tf.constant(range(10), shape=[1, 10])
# b = tf.constant(range(100), shape=[2, 5, 10])

# print(tf.reduce_mean(b, axis = 1))
# print(tf.keras.backend.bias_add(b, a))


# mask = tf.math.logical_not(tf.math.equal(a, 1))
# mask = tf.cast(mask, dtype=a.dtype)
# mask = tf.tile(tf.expand_dims(mask, axis = -1), [1, 1, 10])
# print(mask)
# print(b * mask)
# print(tf.shape(a)[0])

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# a = tf.constant(range(100), shape=[2, 5, 10], dtype=tf.float32)
# cell = tf.keras.layers.LSTM(
#         units = 10,
#         dropout = 1,
#         return_sequences = True,
#         return_state = True,
#         stateful=True,
#         name = 'backward_Encoder')
# bi_cell = tf.keras.layers.Bidirectional(cell)
# init = [tf.zeros((2, 10)), tf.zeros((2, 10))]
# # outputs, hidden, states = cell(a, initial_state=[tf.zeros((2, 10)), tf.zeros((2, 10))])
# # print(cell.states)
# # print(x)
# outputs, fw_hidden, fw_states, bw_hidden, bw_states = bi_cell(a, initial_state= init + init)
# optimizer = tf.keras.optimizers.Adam()

# optimizer.apply_gradients(zip(100, bi_cell.variables))
# print(outputs)
# print(fw_hidden)
# print(fw_states)

a = [1,2,3]
print(*a)
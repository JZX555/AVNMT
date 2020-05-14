import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import numpy as np
import os

import argparse

import core_seq2seq_model
import core_AVNMT_model
import core_data_SRCandTGT
import data_pipeline
from hyper_and_conf import conf_metrics, hyper_param, conf_fn

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print(tf.__version__)
print(np.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

DATA_PATH = 'C:/Users/50568/Desktop/我/maching cafe/unmt/corpus'
hp = hyper_param.HyperParam('large')
PAD_ID_int64 = tf.cast(hp.PAD_ID, tf.int64)

def baseline_evas():
    DataManager = core_data_SRCandTGT.DatasetManager([DATA_PATH + "/europarl-v7.fr-en.fr"],
                                [DATA_PATH + "/europarl-v7.fr-en.en"],
                                batch_size=hp.batch_size,
                                SOS_ID=hp.SOS_ID,
                                PAD_ID=hp.PAD_ID,
                                EOS_ID=hp.EOS_ID,
                                MASK_ID=hp.MASK_ID,
                                UNK_ID=hp.UNK_ID,
                                max_length=hp.max_sequence_length)

    test = DataManager.get_raw_test_dataset()
    # train = train.padded_batch(hp.batch_size, 
    #                            padded_shapes = ([None], [None]), 
    #                            padding_values = (PAD_ID_int64, PAD_ID_int64),
    #                            drop_remainder = True)#.shuffle(DataManager.get_train_size())
    test = data_pipeline._batch_examples(test, hp.batch_size, hp.max_sequence_length)

    # test = DataManager.get_raw_test_dataset()

    model = core_seq2seq_model.Seq2Seq(vocabulary = hp.vocabulary_size,
                enc_units = hp.units,
                dec_units = hp.units,
                batch_size = hp.batch_size,
                embedded_size = hp.embedding_size,
                output_size = hp.vocabulary_size,
                drop_out = hp.dropout,
                max_seq_len = hp.max_sequence_length,
                SOS_ID=hp.SOS_ID,
                PAD_ID=hp.PAD_ID,
                EOS_ID=hp.EOS_ID,
                MASK_ID=hp.MASK_ID,
                UNK_ID=hp.UNK_ID,
                name='baseline_model')  
    
    model.build(None)

    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.lr)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_path = './model_checkpoint\ckpt-57'
    checkpoint.restore(checkpoint_path)
    print('restore {} successful!'.format(checkpoint_path))

    bleu = 0.0
    accuracy = 0.0
    cnt = 0

    for (batch, (x, y)) in enumerate(test):  
        pre = model.inference(x)

        
        tmp_bleu, _= conf_metrics.approx_bleu(y, pre)
        tmp_accuracy, _ = conf_metrics.padded_accuracy(y, pre)
        tmp_accuracy = tf.reduce_mean(tmp_accuracy)

        bleu += tmp_bleu
        accuracy += tmp_accuracy

        cnt += 1
    
    print('bleu on test dataset is: {}'.format(bleu / cnt))
    print('accuracy on test dataset is: {}'.format(accuracy / cnt))

def AVNMT_eva(checkpoint_id):
    DataManager = core_data_SRCandTGT.DatasetManager([DATA_PATH + "/europarl-v7.fr-en.fr"],
                                [DATA_PATH + "/europarl-v7.fr-en.en"],
                                batch_size=hp.batch_size,
                                SOS_ID=hp.SOS_ID,
                                PAD_ID=hp.PAD_ID,
                                EOS_ID=hp.EOS_ID,
                                MASK_ID=hp.MASK_ID,
                                UNK_ID=hp.UNK_ID,
                                max_length=hp.max_sequence_length)

    # train = DataManager.get_raw_train_dataset(shuffle=False)
    # # train = train.padded_batch(hp.batch_size, 
    # #                            padded_shapes = ([None], [None]), 
    # #                            padding_values = (PAD_ID_int64, PAD_ID_int64),
    # #                            drop_remainder = True)#.shuffle(DataManager.get_train_size())

    test = DataManager.get_raw_test_dataset()
    test = data_pipeline._batch_examples(test, hp.batch_size, hp.max_sequence_length)

    model = core_AVNMT_model.AVNMT(vocabulary = hp.vocabulary_size,
                enc_units = hp.units,
                dec_units = hp.units,
                vae_units = hp.units // 4,
                batch_size = hp.batch_size,
                embedded_size = hp.embedding_size,
                output_size = hp.vocabulary_size,
                drop_out = hp.dropout,
                max_seq_len = hp.max_sequence_length,
                SOS_ID=hp.SOS_ID,
                PAD_ID=hp.PAD_ID,
                EOS_ID=hp.EOS_ID,
                MASK_ID=hp.MASK_ID,
                UNK_ID=hp.UNK_ID,
                name='AVNMT')
    for (batch, (x, y)) in enumerate(test):  
        model((x, y))
        break

    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.lr)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_path = './model_checkpoint\ckpt-' + checkpoint_id
    checkpoint.restore(checkpoint_path)
    print('restore {} successful!'.format(checkpoint_path))

    bleu = 0.0
    accuracy = 0.0
    cnt = 0

    print('begin eva!')
    for (batch, (x, y)) in enumerate(test):  
        pre = model.inference(x)
        
        tmp_bleu, _= conf_metrics.approx_bleu(y, pre)
        tmp_accuracy, _ = conf_metrics.padded_accuracy(y, pre)
        tmp_accuracy = tf.reduce_mean(tmp_accuracy)

        bleu += tmp_bleu
        accuracy += tmp_accuracy

        cnt += 1
    
    print('bleu on test dataset is: {}'.format(bleu / cnt))
    print('accuracy on test dataset is: {}'.format(accuracy / cnt))

if __name__ == '__main__':
    # baseline_eva()
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', '-i', type = str ,default = '1', help='the index of the checkpoint')
    args = parser.parse_args()

    AVNMT_eva(args.idx)
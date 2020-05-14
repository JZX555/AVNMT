import tensorflow as tf
import numpy as np
import os
import core_seq2seq_model
import core_data_SRCandTGT
import data_pipeline
from hyper_and_conf import conf_metrics, hyper_param, conf_fn

# import core_model_initializer as init

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print(tf.__version__)
print(np.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

DATA_PATH = 'C:/Users/50568/Desktop/我/maching cafe/unmt/corpus'
RESTORE = True
EPOCHS = 2

hp = hyper_param.HyperParam('large')
PAD_ID_int64 = tf.cast(hp.PAD_ID, tf.int64)

def main():
    DataManager = core_data_SRCandTGT.DatasetManager([DATA_PATH + "/europarl-v7.fr-en.fr"],
                                [DATA_PATH + "/europarl-v7.fr-en.en"],
                                batch_size=hp.batch_size,
                                SOS_ID=hp.SOS_ID,
                                PAD_ID=hp.PAD_ID,
                                EOS_ID=hp.EOS_ID,
                                MASK_ID=hp.MASK_ID,
                                UNK_ID=hp.UNK_ID,
                                max_length=hp.max_sequence_length)

    train = DataManager.get_raw_train_dataset(shuffle=False)
    # train = train.padded_batch(hp.batch_size, 
    #                            padded_shapes = ([None], [None]), 
    #                            padding_values = (PAD_ID_int64, PAD_ID_int64),
    #                            drop_remainder = True)#.shuffle(DataManager.get_train_size())
    train = data_pipeline._batch_examples(train, hp.batch_size, hp.max_sequence_length)

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

    print('initial optimizer')
    # optimizer = tf.keras.optimizers.SGD(learning_rate=hp.lr)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.lr)
    # optimizer = init.get_optimizer(init_lr=hp.lr)

    summary_writer = tf.summary.create_file_writer(hp.model_summary_dir)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_dir = './model_checkpoint'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    if RESTORE:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print('restore {} successful!'.format(tf.train.latest_checkpoint(checkpoint_dir)))

    step = 0
    total_loss = 0
    train_accuarcys = []

    print('begin training!')
    for i in range(hp.epoch_num):
        for (batch, (x, y)) in enumerate(train):  
            if step < 295001:
                step += 1
                continue

            with tf.GradientTape() as model_tape:
                logits = model((x, y))
                batch_loss = model.get_loss(y, logits)
            
            res = model.get_predict_with_logits(logits)

            total_loss += batch_loss       
            step += 1
            model_gradients = model_tape.gradient(batch_loss, model.variables)      
            optimizer.apply_gradients(zip(model_gradients, model.variables))

            batch_accuracy, _ = conf_metrics.padded_accuracy(y, res)
            batch_accuracy = tf.reduce_mean(batch_accuracy)
            batch_bleu, _ = conf_metrics.approx_bleu(y, res)

            if batch % 10 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar('accuracy', batch_accuracy, step)
                    tf.summary.scalar('approx_bleu', batch_bleu, step)
                    tf.summary.scalar('loss', batch_loss, step)
                    summary_writer.flush()

            if batch % 100 == 0:
                print('\n-------------------------------')
                for j in range(np.shape(res)[0]):
                    print('src {} is:{}'.format(j, DataManager.decode(x[j])))
                    print('pre {} is:{}'.format(j, DataManager.decode(res[j])))
                    print('tgt {} is:{}'.format(j, DataManager.decode(y[j])))
                print('Epoch {} Batch {} step{} Loss {}'.format(i,
                                                        batch,
                                                        step,
                                                        batch_loss))
            
            if step % 5000 == 1:
                checkpoint.save(file_prefix = checkpoint_prefix)

# def main():
#     # num_gpus = tf.config.experimental.list_physical_devices('GPU')
#     num_gpus = conf_fn.get_available_gpus()
#     if num_gpus == 0:
#         devices = ["device:CPU:0"]
#     else:
#         devices = ["device:GPU:%d" % i for i in range(num_gpus)]
#     strategy = tf.distribute.MirroredStrategy()
#     with strategy.scope():
#         train_dataset = init.train_input()
#         optimizer = init.raw_optimizer()
#         train_model = init.train_baseline()
#         # train_model.load_weights(
#         #     tf.train.latest_checkpoint("./model_checkpoint/"))
#         train_model.compile(optimizer=optimizer)
#     callbacks = init.get_callbacks()
#     train_model.summary()
#     train_model.fit(train_dataset, epochs=100, verbose=1, callbacks=callbacks)

if __name__ == '__main__':
    main()
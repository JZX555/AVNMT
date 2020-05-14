import tensorflow as tf
import numpy as np
import os
import core_AVNMT_model
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
OUT_DIR = './out'

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

    print('initial optimizer')
    # optimizer = tf.keras.optimizers.SGD(learning_rate=hp.lr)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.lr)
    # optimizer = init.get_optimizer(init_lr=hp.lr)

    D_optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp.lr)
    G_optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp.lr)

    summary_writer = tf.summary.create_file_writer(hp.model_summary_dir)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_dir = './model_checkpoint'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    if RESTORE:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        model.set_lambda_vae(0.6)
        print('restore {} successful!'.format(tf.train.latest_checkpoint(checkpoint_dir)))

    step = 0

    print('begin training!')
    for i in range(hp.epoch_num):
        for (batch, (x, y)) in enumerate(train):  
            if RESTORE and step < 295001:
                step += 1
                continue

            with tf.GradientTape(persistent=True) as model_tape:
                logits, real, fake = model((x, y), 
                                        teacher_force_rate = 0.5,
                                        sample_force_rate = 0.6)
                model_loss, logits_loss, vae_loss, D_loss, G_loss = model.get_model_loss(y, logits, real, fake)
            
            res = model.get_predict_with_logits(logits)
    
            step += 1
            model_gradients = model_tape.gradient(model_loss , model.variables)      
            optimizer.apply_gradients(zip(model_gradients, model.variables))

            D_gradients = model_tape.gradient(D_loss, model.variables)
            D_optimizer.apply_gradients(zip(D_gradients, model.variables))
            model.clip_discriminator()

            G_gradients = model_tape.gradient(G_loss, model.variables)
            G_optimizer.apply_gradients(zip(G_gradients, model.variables))

            batch_accuracy, _ = conf_metrics.padded_accuracy(y, res)
            batch_accuracy = tf.reduce_mean(batch_accuracy)
            batch_bleu, _ = conf_metrics.approx_bleu(y, res)

            if batch % 10 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar('accuracy', batch_accuracy, step)
                    tf.summary.scalar('approx_bleu', batch_bleu, step)
                    tf.summary.scalar('logits_loss', logits_loss, step)
                    tf.summary.scalar('vae_loss', vae_loss, step)
                    tf.summary.scalar('D_loss', D_loss, step)
                    tf.summary.scalar('G_loss', G_loss, step)
                    tf.summary.scalar('total_loss', logits_loss + vae_loss + D_loss + G_loss, step)
                    summary_writer.flush()

            if batch % 100 == 0:
                with open(OUT_DIR, 'a+', encoding="utf-8") as f:
                    f.write('\n-------------------------------\n')
                    for j in range(np.shape(res)[0]):
                        f.write('src {} is:{}\n'.format(j, DataManager.decode(x[j])))
                        f.write('pre {} is:{}\n'.format(j, DataManager.decode(res[j])))
                        f.write('tgt {} is:{}\n'.format(j, DataManager.decode(y[j])))
                    f.write('Epoch {} Batch {} step{} logits_loss {} vae_loss{} D_loss{} G_loss{}\n'.format(i,
                                                            batch,
                                                            step,
                                                            logits_loss,
                                                            vae_loss,
                                                            D_loss,
                                                            G_loss))
                    f.write('real is: {}; fake is: {}; bleu is: {}; accuracy is: {}\n'.format(real,
                                                            fake,
                                                            batch_bleu,
                                                            batch_accuracy))                                                    
            
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
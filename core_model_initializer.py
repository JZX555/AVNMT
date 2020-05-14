# encoding=utf-8
import sys
from hyper_and_conf import hyper_param as hyperParam
from hyper_and_conf import hyper_train, hyper_optimizer, conf_metrics, conf_fn
import core_seq2seq_model
import core_data_SRCandTGT
from tensorflow.python.client import device_lib
import tensorflow as tf
import data_pipeline
import os
DATA_PATH = 'C:/Users/50568/Desktop/我/maching cafe/unmt'
SYS_PATH = sys.path[1]
cwd = os.getcwd()
src_data_path = [DATA_PATH + "/corpus/europarl-v7.fr-en.fr"]
tgt_data_path = [DATA_PATH + "/corpus/europarl-v7.fr-en.en"]
GPUS = conf_fn.get_available_gpus()
GPUS = GPUS if GPUS > 1 else 1


def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'CPU'])


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])


def cpus_device():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']


def gpus_device():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


gpu = get_available_gpus()
TRAIN_MODE = 'large' if gpu > 0 else 'test'
# TRAIN_MODE = 'large'

hp = hyperParam.HyperParam(TRAIN_MODE, gpu=get_available_gpus())
PAD_ID_int64 = tf.cast(hp.PAD_ID, tf.int64)
PAD_ID_float32 = tf.cast(hp.PAD_ID, tf.float32)

data_manager = core_data_SRCandTGT.DatasetManager(
    src_data_path,
    tgt_data_path,
    batch_size=hp.batch_size,
    SOS_ID=hp.SOS_ID,
    PAD_ID=hp.PAD_ID,
    EOS_ID=hp.EOS_ID,
    MASK_ID=hp.MASK_ID,
    UNK_ID=hp.UNK_ID,
    max_length=hp.max_sequence_length)

# train_dataset, val_dataset, test_dataset = data_manager.prepare_data()


def get_hp():
    return hp


def backend_config():
    config = tf.compat.v1.ConfigProto()
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    config.intra_op_parallelism_threads = 4
    config.inter_op_parallelism_threads = 4
    # # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    # config.gpu_options.per_process_gpu_memory_fraction = 0.999
    # config.allow_soft_placement = True

    return config


def input_fn(flag="TRAIN", shuffle=False):
    if flag == "VAL":
        dataset = data_manager.get_raw_val_dataset()
    else:
        if flag == "TEST":
            dataset = data_manager.get_raw_test_dataset()
        else:
            if flag == "TRAIN":
                dataset = data_manager.get_raw_train_dataset(shuffle = False)
            else:
                assert ("data error")
    return dataset


def map_data_for_feed(x, y):
    # return ((x, y), (x, y, x, y))
    return (x, y)


def map_data_for_translation(x, y):
    # return ((x, y), y)
    return ((x, y), )


def map_data_for_eval(x, y):
    return x, y


def pad_sample(dataset, batch_size):
    # dataset = dataset.shuffle(2000000, reshuffle_each_iteration=True)
    dataset = dataset.padded_batch(
        batch_size,
        (
            [hp.max_sequence_length],  # source vectors of unknown size
            [hp.max_sequence_length]),  # target vectors of unknown size
        (PAD_ID_int64, PAD_ID_int64),
        drop_remainder=True)
    return dataset


def dataset_prepross_fn(src, tgt):
    return (
        (src, ),
        tgt,
    )


def train_input(shuffle=False):
    dataset = input_fn('TRAIN', shuffle)
    dataset = dataset.shuffle(20)
    # dataset = data_pipeline._batch_examples(dataset, hp.batch_size, hp.max_sequence_length)
    dataset = pad_sample(dataset, batch_size=hp.batch_size)
    dataset = dataset.map(map_data_for_feed,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def train_supervised_tanslation_input(shuffle=False):
    dataset = input_fn('TRAIN', shuffle)
    dataset = dataset.shuffle(20)
    dataset = pad_sample(dataset, batch_size=hp.batch_size)
    dataset = dataset.map(map_data_for_translation,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def val_input(seq2seq=True):
    dataset = input_fn("VAL")
    dataset = pad_sample(dataset, 1)
    dataset = dataset.map(map_data_for_eval)
    return dataset


def test_input(seq2seq=True):
    dataset = input_fn("TEST")
    dataset = pad_sample(dataset, 1)
    dataset = dataset.map(map_data_for_eval)
    # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def get_external_loss():
    # return conf_metrics.onehot_loss_function
    return hyper_train.Onehot_CrossEntropy(hp.vocabulary_size)


def baseline_structure(training=True):
    if training:
        src_input = tf.keras.layers.Input(shape=[hp.max_sequence_length],
                                        dtype=tf.int64,
                                        name='src_input')
        tgt = tf.keras.layers.Input(shape=[hp.max_sequence_length],
                                    dtype=tf.int64,
                                    name='tgt_input')    

        baseline = core_seq2seq_model.Seq2Seq(vocabulary = hp.vocabulary_size,
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
            
        logits = baseline(
            (src_input, tgt))

        logits = hyper_train.MetricLayer(
            hp.vocabulary_size)([tgt, logits])

        logits = hyper_train.CrossEntropy_layer(
            hp.vocabulary_size, 0.1, name='srclm_loss',
            penalty=0.22)([tgt, logits])

        logits = tf.keras.layers.Lambda(lambda x: x, name="logits")(logits)

        model = tf.keras.Model(inputs=[(src_input, tgt)], outputs=logits)
    
        return model
    else:
        src_input = tf.keras.layers.Input(shape=[None],
                                          dtype=tf.int64,
                                          name='src_input')

        baseline = core_seq2seq_model.Seq2Seq(vocabulary = hp.vocabulary_size,
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

        pre = baseline.inference(src_input)   

        model = tf.keras.Model(inputs=[src_input], outputs=pre)
        return model

# def model_structure(training=True):
#     if training:
#         src_input = tf.keras.layers.Input(shape=[None],
#                                           dtype=tf.int64,
#                                           name='src_input')
#         tgt = tf.keras.layers.Input(shape=[None],
#                                     dtype=tf.int64,
#                                     name='tgt_input')
#         # metric = hyper_train.MetricLayer(hp.vocabulary_size)
#         # loss = hyper_train.CrossEntropy(hp.vocabulary_size, 0.1)

#         boston = unmt_model.UNMT(
#             vocabulary=hp.vocabulary_size,
#             embedding=hp.embedding_size,
#             units=hp.units,
#             # batch_size=hp.batch_size // GPUS,
#             dropout=hp.dropout,
#             heads=hp.heads,
#             max_seq_length=hp.max_sequence_length)
#         # return boston
#         srclm_logits, tgtlm_logits, tgt2src_logits, src2tgt_logits = boston(
#             (src_input, tgt), training=training)
#         # srclm_logits, tgtlm_logits, tgt2src_ids, src2tgt_ids = boston(
#         #     [src_input, tgt], training=training)
#         srclm_logits = hyper_train.CrossEntropy_layer(
#             hp.vocabulary_size, 0.1, name='srclm_loss',
#             penalty=0.22)([src_input, srclm_logits])
#         # _ = hyper_train.Loss_MetricLayer(name='srclm_loss')(srclm_loss)
#         tgtlm_logits = hyper_train.CrossEntropy_layer(hp.vocabulary_size,
#                                                       0.1,
#                                                       name='tgtlm_loss',
#                                                       penalty=0.22)(
#                                                           [tgt, tgtlm_logits])
#         # _ = hyper_train.Loss_MetricLayer(name='srclm_loss')(tgtlm_loss)
#         tgt2src_logits = hyper_train.CrossEntropy_layer(
#             hp.vocabulary_size, 0.1, name='tgt2src_loss',
#             penalty=0.22)([src_input, tgt2src_logits])
#         # _ = hyper_train.Loss_MetricLayer(name='tgt2src_loss')(tgt2src_loss)
#         src2tgt_logits = hyper_train.CrossEntropy_layer(hp.vocabulary_size,
#                                                         0.1,
#                                                         name='src2tgt_loss',
#                                                         penalty=0.34)([
#                                                             tgt, src2tgt_logits
#                                                         ])
#         # _ = hyper_train.Loss_MetricLayer(name='src2src_loss')(src2tgt_loss)
#         # tgt2src_logits = boston.embedding_softmax_layer(tgt2src_ids)
#         # srclm_logits = hyper_train.Quadrugram_BLEU_Metric()(src_input,
#         #                                                     srclm_logits)
#         # tgtlm_logits = hyper_train.Quadrugram_BLEU_Metric()(tgt, tgtlm_logits)
#         src2tgt_logits = hyper_train.MetricLayer(
#             hp.vocabulary_size)([tgt, src2tgt_logits])
#         srclm_logits = hyper_train.Quardru_Bleu_Layer('srclm')(
#             (src_input, srclm_logits))
#         tgt2src_logits = hyper_train.Quardru_Bleu_Layer('tgt2src')(
#             (src_input, tgt2src_logits))
#         tgtlm_logits = hyper_train.Quardru_Bleu_Layer('tgtlm')(
#             (tgt, tgtlm_logits))

#         srclm_logits = tf.keras.layers.Lambda(
#             lambda x: x, name='srclm_logits')(srclm_logits)
#         tgtlm_logits = tf.keras.layers.Lambda(
#             lambda x: x, name='tgtlm_logits')(tgtlm_logits)
#         tgt2src_logits = tf.keras.layers.Lambda(
#             lambda x: x, name='tgt2src_logits')(tgt2src_logits)
#         src2tgt_logits = tf.keras.layers.Lambda(
#             lambda x: x, name='src2tgt_logits')(src2tgt_logits)
#         model = tf.keras.Model(inputs=[src_input, tgt],
#                                outputs=[
#                                    srclm_logits, tgtlm_logits, tgt2src_logits,
#                                    src2tgt_logits
#                                ])
#         # srclm_loss = conf_metrics.onehot_loss_function(
#         #     src_input, srclm_logits, vocab_size=hp.vocabulary_size, pre_sum=True)
#         # model.add_metric(srclm_loss, name='srclm_loss', aggregation='mean')
#         # tgtlm_loss = conf_metrics.onehot_loss_function(
#         #     tgt, tgtlm_logits, vocab_size=hp.vocabulary_size, pre_sum=True)
#         # model.add_metric(tgtlm_loss, name='tgtlm_loss', aggregation='mean')
#         # src2tgt_loss = conf_metrics.onehot_loss_function(
#         #     tgt, src2tgt_logits, vocab_size=hp.vocabulary_size, pre_sum=True)
#         # model.add_metric(src2tgt_loss, name='src2tgt_loss', aggregation='mean')
#         # tgt2src_loss = conf_metrics.onehot_loss_function(
#         #     src_input, tgt2src_logits, vocab_size=hp.vocabulary_size, pre_sum=True)
#         # model.add_metric(tgt2src_loss, name='tgt2src_loss', aggregation='mean')
#         # model.add_loss(srclm_loss * 0.2)
#         # model.add_loss(tgtlm_loss * 0.2)
#         # model.add_loss(src2tgt_loss * 0.4)
#         # model.add_loss(tgt2src_loss * 0.2)
#         # model = tf.keras.Model(
#         #     inputs=[src_input, tgt],
#         #     outputs=[srclm_logits, tgtlm_logits, tgt2src_ids, src2tgt_ids])
#         return model
#     else:
#         src_input = tf.keras.layers.Input(shape=[None],
#                                           dtype=tf.int64,
#                                           name='src_input')
#         boston = unmt_model.UNMT(
#             vocabulary=hp.vocabulary_size,
#             embedding=hp.embedding_size,
#             units=hp.units,
#             # batch_size=hp.batch_size // GPUS,
#             dropout=hp.dropout,
#             heads=hp.heads,
#             max_seq_length=hp.max_sequence_length)
#         # return boston
#         ret = boston((src_input, ), training=training)
#         # outputs, scores = ret["outputs"], ret["scores"]
#         # outputs = hyper_train.MetricLayer(hp.vocabulary_size)([outputs, tgt])
#         model = tf.keras.Model([src_input], ret)
#         return model


def translation_only_model(training=True):
    if training:
        src_input = tf.keras.layers.Input(shape=[None],
                                          dtype=tf.int64,
                                          name='src_input')
        tgt = tf.keras.layers.Input(shape=[None],
                                    dtype=tf.int64,
                                    name='tgt_input')
        boston = unmt_model.UNMT(
            vocabulary=hp.vocabulary_size,
            embedding=hp.embedding_size,
            units=hp.units,
            # batch_size=hp.batch_size // GPUS,
            dropout=hp.dropout,
            heads=hp.heads,
            max_seq_length=hp.max_sequence_length)
        src2tgt_logits, _, _ = boston((src_input, tgt),
                                      training=training,
                                      mode='supervised')
        src2tgt_logits = hyper_train.MetricLayer(
            hp.vocabulary_size)([tgt, src2tgt_logits])
        src2tgt_logits = hyper_train.CrossEntropy_layer(hp.vocabulary_size,
                                                        0.1,
                                                        name='src2tgt_loss')([
                                                            tgt, src2tgt_logits
                                                        ])
        src2tgt_logits = tf.keras.layers.Lambda(
            lambda x: x, name='src2tgt_logits')(src2tgt_logits)

        model = tf.keras.Model(inputs=[src_input, tgt],
                               outputs=[src2tgt_logits])
        # srclm_loss = conf_metrics.onehot_loss_function(
        #     src_input, srclm_logits, vocab_size=hp.vocabulary_size)
        # tgtlm_loss = conf_metrics.onehot_loss_function(
        #     tgt, tgtlm_logits, vocab_size=hp.vocabulary_size)
        # tgt2src_loss = conf_metrics.onehot_loss_function(
        #     src_input, tgt2src_logits, vocab_size=hp.vocabulary_size)
        # model.add_loss(srclm_loss)
        # model.add_loss(tgtlm_loss)
        # model.add_loss(tgt2src_loss)
        return model
        # model = tf.keras.Model(
        #     inputs=[src_input, tgt],
        #     outputs=[srclm_logits, tgtlm_logits, tgt2src_ids, src2tgt_ids])
    else:
        src_input = tf.keras.layers.Input(shape=[None],
                                          dtype=tf.int64,
                                          name='src_input')
        boston = unmt_model.UNMT(
            vocabulary=hp.vocabulary_size,
            embedding=hp.embedding_size,
            units=hp.units,
            # batch_size=hp.batch_size // GPUS,
            dropout=hp.dropout,
            heads=hp.heads,
            max_seq_length=hp.max_sequence_length)
        # return boston
        ret = boston((src_input, ), training=training)
        # outputs, scores = ret["outputs"], ret["scores"]
        # outputs = hyper_train.MetricLayer(hp.vocabulary_size)([outputs, tgt])
        model = tf.keras.Model([src_input], ret)
        return model


# def raw_model():
#     boston = unmt_model.UNMT(
#         vocabulary=hp.vocabulary_size,
#         embedding=hp.embedding_size,
#         units=hp.units,
#         # batch_size=hp.batch_size // GPUS,
#         dropout=hp.dropout,
#         heads=hp.heads,
#         max_seq_length=hp.max_sequence_length)
#     return boston


def train_baseline():
    return baseline_structure(training=True)


def test_baseline():
    return baseline_structure(training=False)


def quardru_bleu(name):
    return hyper_train.Quadrugram_BLEU_Metric(name)


def Uni_bleu():
    return hyper_train.Unigram_BLEU_Metric()


def word_accuracy():
    return hyper_train.Word_Accuracy_Metric()


def word_top5_accuracy():
    return hyper_train.Word_top5_Accuracy_Metric()


def sentence_accuracy():
    return hyper_train.Sentence_Accuracy_Metric()


def wer():
    return hyper_train.Wer_Metric()


def get_optimizer(init_lr=0.0002,
                  num_train_steps=hp.learning_warmup * 80,
                  num_warmup_steps=hp.learning_warmup):
    # return tf.keras.optimizers.Adam(
    #     beta_1=0.1,
    #     beta_2=0.98,
    #     epsilon=1.0e-9,
    # )
    # return hyper_optimizer.AdamWeightDecay(
    #     beta_1=0.1,
    #     beta_2=0.98,
    #     epsilon=1.0e-9,
    #     weight_decay_rate=0.01,
    #     exclude_from_weight_decay=['layer_norm', 'bias'])
    optimizer = hyper_optimizer.create_optimizer(init_lr, num_train_steps,
                                                 num_warmup_steps)
    return optimizer


def raw_optimizer(init_lr=0.0002,
                  num_train_steps=hp.learning_warmup * 80,
                  num_warmup_steps=hp.learning_warmup):
    return hyper_optimizer.AdamWeightDecay(
        beta_1=0.1,
        beta_2=0.98,
        epsilon=1.0e-9,
        weight_decay_rate=0.01,
        exclude_from_weight_decay=['layer_norm', 'bias'])


def get_callbacks():
    lr_fn = hyper_optimizer.warmup_lr_obj(0.0002, hp.learning_warmup,
                                      hp.learning_warmup * 80)
    # # # gradientBoard = hyper_train.GradientBoard(log_dir=hp.model_summary_dir)
    LRschedule = hyper_optimizer.LearningRateScheduler(lr_fn, 0)
    # LRvisualization = hyper_optimizer.LearningRateVisualization()
    TFboard = tf.keras.callbacks.TensorBoard(
        log_dir=hp.model_summary_dir,
        # write_grads=True,
        # histogram_freq=10000,
        # embeddings_freq=1000,
        write_images=True,
        update_freq=500)
    TFchechpoint = tf.keras.callbacks.ModelCheckpoint(
        hp.model_checkpoint_dir + '/model.{epoch:02d}-{loss:.2f}.ckpt',
        monitor='loss',
        save_weights_only=True,
        save_freq=100000,
        verbose=1)
    NaNchecker = tf.keras.callbacks.TerminateOnNaN()
    # ForceLrReduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy',
    #                                                      factor=0.2,
    #                                                      patience=1,
    #                                                      mode='max',
    #                                                      min_lr=0.00001)
    # GradientBoard = hyper_train.GradientBoard(hp.model_summary_dir)
    return [
        LRschedule,
        # LRvisualization,
        TFboard,
        TFchechpoint,
        NaNchecker,
        # GradientBoard
        # ForceLrReduce,
    ]

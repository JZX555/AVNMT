# encoding=utf-8
import os
cwd = os.getcwd()


class HyperParam:
    def __init__(self,
                 mode,
                 gpu=0,
                 vocab=25000,
                 UNK_ID=4,
                 SOS_ID=1,
                 EOS_ID=2,
                 PAD_ID=0,
                 MASK_ID=3):
        self.gpu = gpu
        self.UNK_ID = UNK_ID
        self.SOS_ID = SOS_ID
        self.EOS_ID = EOS_ID
        self.PAD_ID = PAD_ID
        self.MASK_ID = MASK_ID
        self.model_summary_dir = cwd + "/model_summary"
        self.model_weights_dir = cwd + "/model_weights"
        self.model_checkpoint_dir = cwd + "/model_checkpoint"
        try:
            os.makedirs(self.model_weights_dir)
        except OSError:
            pass
        try:
            os.makedirs(self.model_checkpoint_dir)
        except OSError:
            pass
        try:
            os.makedirs(self.model_summary_dir)
        except OSError:
            pass

        self.vocabulary_size = vocab

        if mode == 'test':
            self.test()
        if mode == 'small':
            self.small()
        if mode == 'large':
            self.large()

    def test(self,
             embedding_size=64,
             batch_size=128,
             epoch_num=5,
             units=64,
             heads=2,
             memory_layer=12,
             max_sequence_length=25,
             epoch=1,
             lr=0.001,
             clipping=5,
             learning_warmup=12000,
             dropout=0.4):

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.units = units
        self.heads = heads
        self.memory_layer = memory_layer
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        self.lr = lr
        self.clipping = clipping
        self.learning_warmup = learning_warmup

    # def small(self,
    #           embedding_size=128,
    #           batch_size=8,
    #           epoch_num=5,
    #           num_units=128,
    #           num_heads=4,
    #           num_encoder_layers=2,
    #           num_decoder_layers=2,
    #           max_sequence_length=150,
    #           epoch=1,
    #           lr=2,
    #           clipping=5,
    #           inference_length=150,
    #           data_shuffle=100,
    #           dropout=0.1,
    #           learning_warmup=10000):
    #
    #     self.embedding_size = embedding_size
    #     self.batch_size = batch_size * 2
    #     self.epoch_num = epoch_num
    #     self.num_units = num_units
    #     self.num_heads = num_heads
    #     self.num_encoder_layers = num_encoder_layers
    #     self.num_decoder_layers = num_decoder_layers
    #     self.max_sequence_length = max_sequence_length
    #     self.dropout = dropout
    #     self.lr = lr
    #     self.clipping = clipping
    #     self.data_shuffle = data_shuffle
    #     self.inference_length = inference_length
    #     self.learning_warmup = learning_warmup
    #
    def large(self,
              embedding_size=256,
              batch_size=128,
              epoch_num=5,
              units=512,
              heads=8,
              memory_layer=4,
              max_sequence_length=30,
              epoch=5,
              lr=0.0004,
              clipping=5,
              learning_warmup=10000,
              dropout=0.2):

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.units = units
        self.heads = heads
        self.memory_layer = memory_layer
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        self.lr = lr
        self.clipping = clipping
        self.learning_warmup = learning_warmup

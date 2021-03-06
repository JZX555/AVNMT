# encoding=utf8
from hyper_and_conf import conf_fn as train_conf
from data import data_setentceToByte_helper
import numpy as np
import tensorflow as tf
import six
import random


class DatasetManager():
    def __init__(self,
                 source_data_path,
                 target_data_path,
                 batch_size=32,
                 shuffle=100,
                 num_sample=-1,
                 max_length=50,
                 SOS_ID=1,
                 EOS_ID=2,
                 PAD_ID=0,
                 MASK_ID=3,
                 UNK_ID=4,
                 cross_val=[0.89, 0.1, 0.01],
                 byte_token='@@',
                 word_token=' ',
                 split_token='\n'):
        """Short summary.

        Args:
            source_data_path (type): Description of parameter `source_data_path`.
            target_data_path (type): Description of parameter `target_data_path`.
            num_sample (type): Description of parameter `num_sample`.
            batch_size (type): Description of parameter `batch_size`.
            split_token (type): Description of parameter `split_token`.

        Returns:
            type: Description of returned object.

        """
        self.source_data_path = source_data_path
        self.target_data_path = target_data_path
        self.num_sample = num_sample
        self.batch_size = batch_size
        self.byte_token = byte_token
        self.split_token = split_token
        self.word_token = word_token
        self.SOS_ID = SOS_ID
        self.EOS_ID = EOS_ID
        self.PAD_ID = PAD_ID
        self.MASK_ID = MASK_ID
        self.UNK_ID = UNK_ID
        self.shuffle = shuffle
        self.max_length = max_length
        self.cross_val = cross_val
        assert isinstance(self.cross_val, list) is True
        self.byter = data_setentceToByte_helper.Subtokenizer(
            self.source_data_path + self.target_data_path,
            SOS_ID=self.SOS_ID,
            PAD_ID=self.PAD_ID,
            EOS_ID=self.EOS_ID,
            UNK_ID=self.UNK_ID,
            MASK_ID=self.MASK_ID)
        if train_conf.get_available_gpus() > 0:
            self.cpus = 4 * train_conf.get_available_gpus()
        else:
            self.cpus = 4
        self.tfrecord_generater()

    def corpus_length_checker(self, data=None, re=True):
        self.short_20 = 0
        self.median_50 = 0
        self.long_100 = 0
        self.super_long = 0
        for k, v in enumerate(data):
            v = v.split(self.word_token)
            v_len = len(v)
            if v_len <= 40:
                self.short_20 += 1
            if v_len > 40 and v_len <= 60:
                self.median_50 += 1
            if v_len > 80 and v_len <= 120:
                self.long_100 += 1
            if v_len > 120:
                self.super_long += 1
        if re:
            print("short: %d" % self.short_20)
            print("median: %d" % self.median_50)
            print("long: %d" % self.long_100)
            print("super long: %d" % self.super_long)

    def cross_validation(self,
                         src_path_list,
                         tgt_path_list,
                         validation=0.0,
                         test=0.05):
        print("Cross validation process")
        assert len(src_path_list) == len(tgt_path_list)
        train_path = []
        test_path = []
        train_path_shuffled = []
        val_path = []
        raw_data = []
        shuffle_raw_data = []
        index = 0
        self.data_counter = 0
        self.val_size = 0
        self.test_size = 0
        self.train_size = 0
        for k, v in enumerate(src_path_list):
            src_path = src_path_list[k]
            tgt_path = tgt_path_list[k]
            train_path_word = "./data/train_data_WORD_LEVEL_" + str(index)
            train_path_word_shuffled = "./data/train_data_WORD_LEVEL_shuffled_" + \
                str(index)
            test_path_word = "./data/test_data_WORD_LEVEL_" + str(index)
            val_path_word = "./data/val_data_WORD_LEVEL_" + str(index)
            with tf.io.gfile.GFile(src_path, "r") as f_src:
                src_raw_data = org_src = f_src.readlines()
                print('src statistic')
                self.corpus_length_checker(src_raw_data)
                # random.shuffle(src_raw_data)
                with tf.io.gfile.GFile(tgt_path, "r") as f_tgt:
                    tgt_raw_data = org_tgt = f_tgt.readlines()
                    print('tgt statistic')
                    self.corpus_length_checker(tgt_raw_data)
                    raw_data += list(zip(org_src, org_tgt))
                    random.shuffle(src_raw_data)
                    random.shuffle(tgt_raw_data)
                    shuffle_raw_data += list(zip(src_raw_data, tgt_raw_data))
                    f_src.close()
                self.data_counter = len(raw_data)
                print(self.data_counter)
                self.val_size = 5000
                self.test_size = 5000
                self.train_size = self.data_counter - 10000
                f_src.close()

                def writer(path, data, byte=False):
                    if tf.io.gfile.exists(path) is not True:
                        with tf.io.gfile.GFile(path, "w") as f:
                            for w in data:
                                if len(w) >= 0:
                                    f.write(w[0].rstrip() + self.byte_token
                                            + w[1])
                            f.close()
                    print("File exsits: {}".format(path))

                if len(src_path_list) == 1 or len(raw_data) > 30000:
                    writer(train_path_word_shuffled,
                           shuffle_raw_data[:self.train_size])
                    writer(train_path_word, raw_data[:self.train_size])
                    # writer(val_path, raw_data[:128])
                    writer(val_path_word,
                           raw_data[self.train_size:self.train_size + 5000])
                    writer(test_path_word, raw_data[self.train_size + 10000:])
                    train_path_shuffled.append(train_path_word_shuffled)
                    train_path.append(train_path_word)
                    val_path.append(val_path_word)
                    test_path.append(val_path_word)
                    index += 1
                    raw_data = []
                else:
                    pass
            print('Total data {0}'.format(self.data_counter))
            print(('Train {0}, Validation {1}, Test {2}'.format(
                self.train_size, self.val_size, self.test_size)))
        return train_path_shuffled, train_path, test_path, val_path

    def tfrecord_generater(self):
        prefix_train = "./data/train_TFRecord_"
        prefix_val = "./data/val_TFRecord_"
        prefix_test = "./data/test_TFRecord_"
        if len(self.cross_val) == 2:
            # train_por = self.cross_val[0]
            val_por = 0
            test_por = self.cross_val[1]
        else:
            # train_por = self.cross_val[0]
            val_por = self.cross_val[1]
            test_por = self.cross_val[2]

        train_path_shuffled, train_path, test_path, val_path = self.cross_validation(
            self.source_data_path,
            self.target_data_path,
            validation=val_por,
            test=test_por)

        def all_exist(filepaths):
            """Returns true if all files in the list exist."""
            for fname in filepaths:
                if not tf.io.gfile.exists(fname):
                    return False
            return True

        def txt_line_iterator(path):
            with open(path, encoding='UTF-8') as f:
                for line in f.readlines():
                    yield line.strip()

        def dict_to_example(dictionary):
            """Converts a dictionary of string->int to a tf.Example."""
            features = {}
            for k, v in six.iteritems(dictionary):
                features[k] = tf.train.Feature(int64_list=tf.train.Int64List(
                    value=v))
            return tf.train.Example(features=tf.train.Features(
                feature=features))

        def tfrecorder(path_list, prefix):
            if len(path_list) > 0:
                tfr_path = [prefix + str(k) for k, _ in enumerate(path_list)]
            if all_exist(tfr_path):
                print("TFRecord already exists!")
            else:
                options = tf.io.TFRecordOptions(
                    compression_type='GZIP')
                writers = [tf.io.TFRecordWriter(
                    fname, options=options) for fname in tfr_path]
                for i, _ in enumerate(writers):
                    for k, v in enumerate(txt_line_iterator(path_list[i])):
                        src, tgt = v.split(self.byte_token)
                        example = dict_to_example({
                            "src":
                            self.byter.encode(src, add_eos=True),
                            "tgt":
                            self.byter.encode(tgt, add_eos=True)
                        })
                        writers[i].write(example.SerializeToString())

                for writer in writers:
                    writer.close()
            return tfr_path

        self.train_tfr = tfrecorder(train_path, prefix_train)
        self.train_tfr_shuffled = tfrecorder(
            train_path_shuffled,
            prefix_train+'shuffled_')
        print("Train TFRecord generated {}".format(self.train_tfr))
        self.val_tfr = tfrecorder(val_path, prefix_val)
        print("Val TFRecord generated {}".format(self.val_tfr))
        self.test_tfr = tfrecorder(test_path, prefix_test)
        print("Test TFRecord generated {}".format(self.test_tfr))
        print("All TFRecord generated")

    def encode(self, string):
        return self.byter.encode(string)

    def decode(self, string):
        return self.byter.decode(string)

    def create_dataset(self, data_path, max_length=0):
        if max_length == 0:
            max_length = self.max_length

        def _parse_example(serialized_example):
            """Return inputs and targets Tensors from a serialized tf.Example."""
            data_fields = {
                "src": tf.io.VarLenFeature(tf.int64),
                "tgt": tf.io.VarLenFeature(tf.int64)
            }
            parsed = tf.io.parse_single_example(serialized_example,
                                                data_fields)
            src = tf.sparse.to_dense(parsed["src"])
            tgt = tf.sparse.to_dense(parsed["tgt"])
            return src, tgt

        def _filter_max_length(example, max_length=256):
            return tf.logical_and(
                tf.size(example[0]) <= max_length,
                tf.size(example[1]) <= max_length)

        dataset = tf.data.TFRecordDataset(
            data_path, compression_type='GZIP')
        dataset = dataset.map(_parse_example,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.filter(lambda x, y: _filter_max_length(
            (x, y), self.max_length))
        return dataset

    def get_raw_train_dataset(self, shuffle=True):
        if shuffle:
            return self.create_dataset("./data/train_TFRecord_shuffled_0")
        else:
            return self.create_dataset("./data/train_TFRecord_0")

    def get_raw_val_dataset(self):
        return self.create_dataset("./data/val_TFRecord_0")

    def get_raw_test_dataset(self):
        return self.create_dataset("./data/test_TFRecord_0")

    def get_train_size(self):
        return self.train_size

    def get_val_size(self):
        return self.val_size

    def get_test_size(self):
        return self.test_size


# tf.enable_eager_execution()
# DATA_PATH = '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng'
# sentenceHelper = DatasetManager([DATA_PATH + "/europarl-v7.fr-en.fr"],
#                                 [DATA_PATH + "/europarl-v7.fr-en.en"],
#                                 batch_size=16,
#                                 shuffle=100)
# # # # # a, b, c = sentenceHelper.prepare_data()
# # # # a, b, c = sentenceHelper.post_process()
# dataset = sentenceHelper.get_raw_train_dataset()
# # for i, e in enumerate(a):
# #     print(e[0])
# #     print(i)
# #     sentenceHelper.byter.decode(e[0].numpy())
# #     break
#
#
# def dataset_prepross_fn(src, tgt):
#     return (src, tgt), tgt
#
#
# dataset = dataset.map(dataset_prepross_fn, num_parallel_calls=12)
# dataset = dataset.padded_batch(
#     1,
#     padded_shapes=(
#         (
#             tf.TensorShape([None]),  # source vectors of unknown size
#             tf.TensorShape([None]),  # target vectors of unknown size
#         ),
#         tf.TensorShape([None])),
#     drop_remainder=True)
# for i in range(5):
#     d = dataset.make_one_shot_iterator()
#     d = d.get_next()

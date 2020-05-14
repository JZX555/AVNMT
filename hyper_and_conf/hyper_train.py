# encoding=utf8
import os
import six
import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
import hyper_and_conf.conf_metrics as conf_metrics
import hyper_and_conf.hyper_fn as hyper_fn
import numpy as np
from tensorflow.python.keras.losses import Loss
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.summary import summary as tf_summary
GPUS = len(tf.config.experimental.list_physical_devices('GPU'))
GPUS = GPUS if GPUS > 0 else 1


class Unigram_BLEU_Metric(tf.keras.metrics.Mean):
    def __init__(self):
        super(Unigram_BLEU_Metric, self).__init__()
        # self.total = self.add_weight(name='UnigramBLEU', initializer='zeros')
        # self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        value, _ = conf_metrics.approx_unigram_bleu(y_true, y_pred)
        self.count.assign_add(1)
        self.total.assign_add(tf.reduce_mean(value))

    # def result(self):
    #     return math_ops.div_no_nan(self.total, self.count)
    #
    # def reset_states(self):
    #     self.total.assign(0.)
    #     self.count.assign(0)


class Quadrugram_BLEU_Metric(tf.keras.metrics.Mean):
    def __init__(self, name):
        super(Quadrugram_BLEU_Metric, self).__init__(name=name)
        # self.total = self.add_weight(name='QuadrugramBLEU',
        #                              initializer='zeros')
        # self.count = self.add_weight('count', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        value, _ = conf_metrics.approx_bleu(y_true, y_pred)
        self.count.assign_add(1)
        self.total.assign_add(tf.reduce_mean(value))

    # def result(self):
    #     return math_ops.div_no_nan(self.total, self.count)
    #
    # def reset_states(self):
    #     self.total.assign(0.)
    #     self.count.assign(0)


class Word_Accuracy_Metric(tf.keras.metrics.Mean):
    def __init__(self, name):
        super(Word_Accuracy_Metric, self).__init__(name=name)
        # self.total = self.add_weight(name='WordAccuracy', initializer='zeros')
        # self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        value, _ = conf_metrics.padded_accuracy(y_true, y_pred)
        self.count.assign_add(1)
        self.total.assign_add(tf.reduce_mean(value))

    # def result(self):
    #     return math_ops.div_no_nan(self.total, self.count)
    #
    # def reset_states(self):
    #     self.total.assign(0.)
    #     self.count.assign(0)


class Sentence_Accuracy_Metric(tf.keras.metrics.Mean):
    def __init__(self, name):
        super(Sentence_Accuracy_Metric, self).__init__(name=name)
        # self.total = self.add_weight(name='SentenceAccuracy',
        #                              initializer='zeros')
        # self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        value, _ = conf_metrics.padded_sequence_accuracy(y_true, y_pred)
        self.count.assign_add(1)
        self.total.assign_add(tf.reduce_mean(value))

    # def result(self):
    #     return math_ops.div_no_nan(self.total, self.count)
    #
    # def reset_states(self):
    #     self.total.assign(0.)
    #     self.count.assign(0)


class Word_top5_Accuracy_Metric(tf.keras.metrics.Mean):
    def __init__(self, name):
        super(Word_top5_Accuracy_Metric, self).__init__(name=name)
        self.total = self.add_weight(name='WordTop5Accuracy',
                                     initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        value, _ = conf_metrics.padded_accuracy_top5(y_true, y_pred)
        self.count.assign_add(1)
        self.total.assign_add(tf.reduce_mean(value))

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)

    def reset_states(self):
        self.total.assign(0.)
        self.count.assign(0)


class Wer_Metric(tf.keras.metrics.Mean):
    def __init__(self, name):
        super(Wer_Metric, self).__init__(name=name)
        self.total = self.add_weight(name='WER', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        value, _ = conf_metrics.wer_score(y_true, y_pred)
        self.count.assign_add(1)
        self.total.assign_add(tf.reduce_mean(value))

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)

    def reset_states(self):
        self.total.assign(0.)
        self.count.assign(0)


class Loss_Metric(tf.keras.metrics.Mean):
    def __init__(self, name='custom_loss'):
        super(Loss_Metric, self).__init__(name=name)
        self.total = self.add_weight(name='loss', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')
        # self.total = 0
        # self.count = 0

    def update_state(self, value, sample_weight=None):
        value = tf.py_function(lambda x: tf.cast(x, tf.float32), [value],
                               tf.float32)
        # value, _ = conf_metrics.wer_score(y_true, y_pred)
        # if sample_weight is not None:
        #     sample_weight = tf.cast(sample_weight, 'float32')
        #     values = tf.multiply(values, sample_weight)
        # self.count = tf.add(self.count, 1)
        # self.total = tf.add(self.total, value)
        self.count += 1
        self.total += value

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total = 0
        self.count = 0


class Onehot_CrossEntropy(Loss):
    def __init__(self, vocab_size, mask_id=0, smoothing=0.1):

        super(Onehot_CrossEntropy, self).__init__(name="Onehot_CrossEntropy")
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.smoothing = smoothing

    def call(self, true, pred):
        # batch_size = tf.shape(true)[0]
        # true = tf.reshape(true, [batch_size, -1])
        loss = conf_metrics.onehot_loss_function(true=true,
                                                 pred=pred,
                                                 mask_id=self.mask_id,
                                                 smoothing=self.smoothing,
                                                 vocab_size=self.vocab_size,
                                                 pre_sum=False)
        return loss


class CrossEntropy_layer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, label_smoothing, penalty=1, name='custom', log_loss=True, **kwargs):
        super(CrossEntropy_layer, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.penalty = penalty
        self.log_loss = log_loss
        if self.log_loss:
            self.mean_fn = tf.keras.metrics.Mean(name)

    def build(self, input_shape):
        super(CrossEntropy_layer, self).build(input_shape)

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "label_smoothing": self.label_smoothing,
        }

    def call(self, inputs):
        targets, logits = inputs[0], inputs[1]
        loss = conf_metrics.onehot_loss_function(
            targets,
            logits,
            smoothing=self.label_smoothing,
            vocab_size=self.vocab_size, pre_sum=True)
        self.add_loss(loss * self.penalty)
        if self.log_loss:
            m = self.mean_fn(loss)
            self.add_metric(m)
        return logits


class Quardru_Bleu_Layer(tf.keras.layers.Layer):
    def __init__(self, n):
        super(Quardru_Bleu_Layer, self).__init__()
        self.n = n

    def build(self, input_shape):
        """"Builds metric layer."""
        self.mean = tf.keras.metrics.Mean(self.n)
        super(Quardru_Bleu_Layer, self).build(input_shape)

    def get_config(self):
        return {}

    def call(self, inputs):
        targets, logits = inputs[0], inputs[1]
        # TODO(guptapriya): Remove this check when underlying issue to create metrics
        # with dist strat in cross replica context is fixed.
        # if tf.distribute.has_strategy(
        # ) and not tf.distribute.in_cross_replica_context():
        m = self.mean(*conf_metrics.approx_bleu(targets, logits))
        self.add_metric(m)
        # else:
        #     for mean, fn in self.metric_mean_fns:
        #         m = mean(*fn(logits, targets))
        #         self.add_metric(m)
        return logits


class Loss_MetricLayer(tf.keras.layers.Layer):
    """Custom a layer of metrics for Transformer model."""

    def __init__(self, name='custom_loss'):
        self.mean_fn = tf.keras.metrics.Mean(name)
        super(Loss_MetricLayer, self).__init__(name=name)

    def build(self, input_shape):
        """"Builds metric layer."""
        super(Loss_MetricLayer, self).build(input_shape)

    def get_config(self):
        return {"vocab_size": self.vocab_size}

    def call(self, inputs):
        # TODO(guptapriya): Remove this check when underlying issue to create metrics
        # with dist strat in cross replica context is fixed.
        # if tf.distribute.has_strategy(
        # ) and not tf.distribute.in_cross_replica_context():
        m = self.mean_fn(inputs)
        self.add_metric(m)
        # else:
        #     for mean, fn in self.metric_mean_fns:
        #         m = mean(*fn(logits, targets))
        #         self.add_metric(m)
        return inputs


class MetricLayer(tf.keras.layers.Layer):
    """Custom a layer of metrics for Transformer model."""

    def __init__(self, vocab_size, criteron='all'):
        super(MetricLayer, self).__init__()
        self.vocab_size = vocab_size
        self.metric_mean_fns = []

    def build(self, input_shape):
        """"Builds metric layer."""
        self.metric_mean_fns = [
            (tf.keras.metrics.Mean("approx_4-gram_bleu"),
             conf_metrics.approx_bleu),
            # (tf.keras.metrics.Mean("approx_unigram_bleu"),
            #  conf_metrics.approx_unigram_bleu),
            (tf.keras.metrics.Mean("wer"), conf_metrics.wer_score),
            (tf.keras.metrics.Mean("accuracy"), conf_metrics.padded_accuracy),
            # (tf.keras.metrics.Mean("accuracy_top5"),
            #  conf_metrics.padded_accuracy_top5),
            # (tf.keras.metrics.Mean("accuracy_per_sequence"),
            #  conf_metrics.padded_sequence_accuracy),
        ]
        super(MetricLayer, self).build(input_shape)

    def get_config(self):
        return {"vocab_size": self.vocab_size}

    def call(self, inputs):
        targets, logits = inputs[0], inputs[1]
        # TODO(guptapriya): Remove this check when underlying issue to create metrics
        # with dist strat in cross replica context is fixed.
        # if tf.distribute.has_strategy(
        # ) and not tf.distribute.in_cross_replica_context():
        for mean, fn in self.metric_mean_fns:
            m = mean(*fn(targets, logits))
            self.add_metric(m)
        # else:
        #     for mean, fn in self.metric_mean_fns:
        #         m = mean(*fn(logits, targets))
        #         self.add_metric(m)
        return logits


class Dynamic_LearningRate(Callback):
    def __init__(self,
                 init_lr,
                 num_units,
                 learning_rate_warmup_steps,
                 verbose=0):
        super(Dynamic_LearningRate, self).__init__()
        self.init_lr = init_lr
        self.num_units = num_units
        self.learning_rate_warmup_steps = learning_rate_warmup_steps
        self.verbose = verbose
        self.sess = tf.compat.v1.keras.backend.get_session()
        self._total_batches_seen = 0
        self.current_lr = 0

    def on_train_begin(self, logs=None):
        self.current_lr = hyper_fn.get_learning_rate(
            self.init_lr, self.num_units, self._total_batches_seen,
            self.learning_rate_warmup_steps)
        lr = float(self.current_lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nStart  learning ' 'rate from %s.' % (lr))

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            self.current_lr = hyper_fn.get_learning_rate(
                self.init_lr, self.num_units, self._total_batches_seen,
                self.learning_rate_warmup_steps)
        except Exception:  # Support for old API for backward compatibility
            self.current_lr = self.init_lr
        lr = float(self.current_lr)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            tf.compat.v1.logging.info('\nEpoch %05d: Changing  learning '
                                      'rate to %s.' % (batch + 1, lr))

    def on_batch_end(self, batch, logs=None):
        # path = os.path.join("model_summary", "train")
        # writer = summary_ops_v2.create_file_writer(path)
        # with summary_ops_v2.always_record_summaries():
        #     with writer.as_default():
        #         summary_ops_v2.scalar(
        #             "lr", self.current_lr, step=self._total_batches_seen)
        self._total_batches_seen += 1
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

    # def _log_lr(logs, prefix,step):


class GradientBoard(Callback):
    def __init__(
            self,
            log_dir='logs',
    ):
        super(GradientBoard, self).__init__()

        self.log_dir = log_dir
        # self.histogram_freq = histogram_freq

        self._samples_seen = 0
        self._samples_seen_at_last_write = 0
        self._current_batch = 0
        self._total_batches_seen = 0
        self._total_val_batches_seen = 0

        # A collection of file writers currently in use, to be closed when
        # training ends for this callback. Writers are keyed by the
        # directory name under the root logdir: e.g., "train" or
        # "validation".
        self._writers = {}
        self._train_run_name = 'train'
        self._validation_run_name = 'validation'

        # TensorBoard should only write summaries on the chief when in a
        # Multi-Worker setting.
        self._chief_worker_only = True

    def set_model(self, model):
        """Sets Keras model and writes graph if specified."""
        self.model = model
        with context.eager_mode():
            # self._make_histogram_ops(model)
            self._close_writers()

    def _close_writers(self):
        """Close all remaining open file writers owned by this callback.
    If there are no such file writers, this is a no-op.
    """
        with context.eager_mode():
            for writer in six.itervalues(self._writers):
                writer.close()
            self._writers.clear()

    def _get_writer(self, writer_name):
        """Get a summary writer for the given subdirectory under the logdir.
    A writer will be created if it does not yet exist.
    Arguments:
      writer_name: The name of the directory for which to create or
        retrieve a writer. Should be either `self._train_run_name` or
        `self._validation_run_name`.
    Returns:
      A `SummaryWriter` object.
    """
        if writer_name not in self._writers:
            path = os.path.join(self.log_dir, writer_name)
            writer = summary_ops_v2.create_file_writer_v2(path)
            self._writers[writer_name] = writer
        return self._writers[writer_name]

    def on_batch_end(self, batch, logs=None):
        """Writes scalar summaries for metrics on every training batch.
    Performs profiling if current batch is in profiler_batches.
    Arguments:
      batch: Integer, index of batch within the current epoch.
      logs: Dict. Metric results for this batch.
    """
        # Don't output batch_size and batch number as TensorBoard summaries
        logs = logs or {}
        self._samples_seen += logs.get('size', 1)
        samples_seen_since = self._samples_seen - self._samples_seen_at_last_write
        if samples_seen_since >= 1:
            self._log_gradient(batch)
            self._samples_seen_at_last_write = self._samples_seen
        self._total_batches_seen += 1

    def on_train_end(self, logs=None):
        self._close_writers()

    # def _make_histogram_ops(self, model):
    #     for layer in model.layers[2].layers:
    #         for weight in layer.trainable_weights:
    #             mapped_weight_name = weight.name.replace(':', '_')
    #             with ops.init_scope():
    #                 weight = tf.keras.backend.get_value(weight)
    #             grads = model.optimizer.get_gradients(model.losses[0], weight)
    #
    #             def is_indexed_slices(grad):
    #                 return type(grad).__name__ == 'IndexedSlices'
    #
    #             grads = [
    #                 grad.values if is_indexed_slices(grad) else grad
    #                 for grad in grads
    #             ]
    #             tf_summary.histogram('{}_grad'.format(mapped_weight_name),
    #                                  grads)
    def _log_gradient(self, model):
        weight = self.model.trainable_weights
        with ops.init_scope():
            weight = tf.keras.backend.get_value(weight)
        grads = model.optimizer.get_gradients(model.losses[0], weight)
        grads = tf.norm(grads)
        tf_summary.scalar('{}_grad'.format('grad_norm'),
                          grads)

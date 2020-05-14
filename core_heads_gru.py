# encoding=utf-8
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops import array_ops
from hyper_and_conf import hyper_layer, hyper_fn, hyper_util
import random
import numpy as np
from tensorflow.python.keras.engine.input_spec import InputSpec


class HeadsGRUcell(tf.keras.layers.GRUCell):
    def __init__(self,
                 units,
                 dropout=0.,
                 att=True,
                 name='LSTM',
                 residual=True,
                 external=True,
                 heads=4,
                 implementation=2,
                 **kwargs):
        super(HeadsGRUcell, self).__init__(units,
                                           dropout=dropout,
                                           implementation=implementation,
                                           name=name,
                                           dynamic=True,
                                           reset_after=False,
                                           **kwargs)
        self.units = units
        self.dropout = dropout
        self.context = None
        self.att = att
        self.external = external
        self.heads = heads
        self.implementation = 2
        self.residual = residual

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # self.bias = self.add_weight(shape=(3 * self.units),
        #                   name='bias',
        #                   initializer=self.bias_initializer,
        #                   regularizer=self.bias_regularizer,
        #                   constraint=self.bias_constraint)
        self.q_kernel = self.add_weight(shape=(input_dim, self.units * 3),
                                        name='kernel',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        self.k_kernel = self.add_weight(shape=(input_dim, self.units * 3),
                                        name='kernel',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        self.v_kernel = self.add_weight(shape=(input_dim, self.units * 3),
                                        name='kernel',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        # self.output_z = tf.keras.layers.Dense(self.units,
        #                                       use_bias=False,
        #                                       name="output_z",
        #                                       activation='sigmoid')
        # self.output_h = tf.keras.layers.Dense(self.units,
        #                                       use_bias=False,
        #                                       name="output_h",
        #                                       activation='tanh')
        # self.output_z_norm = hyper_layer.LayerNorm()
        # self.output_r = tf.keras.layers.Dense(self.units,
        #                                       use_bias=False,
        #                                       name="output_z")
        # self.output_h_norm = tf.keras.layers.Dense(self.units,
        #                                            use_bias=False,
        #                                            name="output_h",
        #                                            activation='tanh')

        # self.output_h_norm = hyper_layer.LayerNorm()
        # self.heads_filter = hyper_layer.NormBlock(hyper_layer.Feed_Forward_Network(self.units,
        #                                                                            self.dropout,
        #                                                                            activation_filter='relu'), self.dropout)
        self.r_heads_filter = tf.nn.sigmoid
        # self.z_heads_filter = tf.nn.sigmoid
        # self.h_heads_filter = tf.nn.relu
        # self.r_heads_filter = tf.keras.layers.Dense(self.units,
        #                                             use_bias=False,
        #                                             activation='sigmoid')
        self.h_heads_filter = tf.keras.layers.Dense(self.units,
                                                    use_bias=False,
                                                    activation='tanh')
        self.z_heads_filter = tf.keras.layers.Dense(self.units,
                                                    use_bias=False,
                                                    activation='sigmoid')
        # self.r_heads_filter = hyper_layer.Feed_Forward_Network(
        #     self.units * 4,
        #     self.dropout,
        #     activation_filter='relu',
        #     activation_output='sigmoid')
        # self.z_heads_filter = hyper_layer.Feed_Forward_Network(
        #     self.units * 4,
        #     self.dropout,
        #     activation_filter='relu',
        #     activation_output='sigmoid')
        # self.z_heads_filter = hyper_layer.Feed_Forward_Network(
        #     self.units * 4,
        #     self.dropout,
        #     activation_filter='relu',
        #     activation_output='sigmoid')
        #
        # self.h_heads_filter = hyper_layer.Feed_Forward_Network(
        #     self.units * 4,
        #     self.dropout,
        #     activation_filter='relu',
        #     activation_output='tanh')

        # self.h_norm = hyper_layer.LayerNorm()
        self.input_norm = hyper_layer.LayerNorm(model="tanh")
        # self.inner_norm = hyper_layer.LayerNorm(model="tanh")
        self.z_layer_norm = hyper_layer.LayerNorm()
        # self.r_layer_norm = hyper_layer.LayerNorm()
        self.hh_layer_norm = hyper_layer.LayerNorm()
        # self.layer_norm_hidden = hyper_layer.LayerNorm()
        super(HeadsGRUcell, self).build(input_shape)

    def call(self, inputs, states, constants, training=False):
        org = inputs
        constants = states + list(constants)
        inputs = self.input_norm(inputs)
        # import pdb; pdb.set_trace()
        h_tm1 = constants[0]  # previous memory
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        if 0. < self.dropout < 1.:
            inputs = inputs * dp_mask[0]
        # inputs = self.split_heads_2to3(inputs, self.heads)a
        # multi_h_tm1 = self.split_heads_2to3(h_tm1, self.heads)
        matrix_x = K.dot(inputs, self.kernel)
        matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
        # matrix_inner = K.dot(h_tm1, self.recurrent_kernel[:, :2 * self.units])
        if self.use_bias:
            # input_bias = self.split_heads_2to3(input_bias)
            # recurrent_bias = self.split_heads_2to3(recurrent_bias)
            matrix_x = K.bias_add(matrix_x, self.bias)
        x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=-1)
        recurrent_z, recurrent_r, recurrent_h = array_ops.split(matrix_inner,
                                                                3,
                                                                axis=-1)
        if constants[1] is not None:
            attention_context = self.heads_attention_wrapper(
                h_tm1, constants, training)
            att_z, att_r, att_h = array_ops.split(attention_context,
                                                  3,
                                                  axis=-1)
            org_z = x_z + recurrent_z + att_z
            # org_z = self.output_z(z)
            # z = self.z_heads_filter(self.z_layer_norm(org_z))
            z = self.z_heads_filter(self.z_layer_norm(org_z))
            if 0. < self.dropout < 1.:
                z = z * dp_mask[0]
            z = z + org_z

            r = x_r + recurrent_r + att_r
            # r = self.r_heads_filter(r)
            r = self.r_heads_filter(r)
            # recurrent_h = K.dot(r * h_tm1,
            #                     self.recurrent_kernel[:, 2 * self.units:])
            org_hh = x_h + r * recurrent_h + att_h
            # hh = self.h_heads_filter(self.hh_layer_norm(org_hh))
            hh = self.h_heads_filter(self.hh_layer_norm(org_hh))
            if 0. < self.dropout < 1.:
                hh = hh * dp_mask[0]
            hh = hh + org_hh
            # hh = self.hh_layer_norm(hh)
        else:
            z = self.z_heads_filter(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            hh = self.h_heads_filter(x_h + r * recurrent_h)
        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh
        # h = tf.reshape(h, [-1, self.units])
        # h = self.heads_filter(h)
        # h = self.heads_filter(h)
        # if training:
        #     h = tf.nn.dropout(h, self.dropout)
        # if 0. < self.dropout < 1.:
        #     h = h * dp_mask[0]
        # org = org + h
        # h = self.heads_filter(self.h_norm(org))
        if training:
            # h = tf.nn.dropout(h,self.dropout)
            h = h * dp_mask[0]
        h = org + h
        # # last = constants[0] + h
        # #
        # h = self.heads_filter(self.layer_norm_filter(org), training)
        # if training:
        #     h = tf.nn.dropout(h, self.dropout)
        # h = self.layer_norm(h + org)
        # c = constants[1] + c
        return h, [h]

    def heads_attention_wrapper(self, inputs, states, training):
        # batch_size = tf.shape(inputs)[0]
        # random_external = random.randint(0, 10)
        # if random_external > 7 and states[2] is not None:
        #     contexts = tf.concat((states[1], states[2]), -1)
        #
        # else:
        #     contexts = tf.concat((states[1] * 2, tf.zeros_like(states[1])), -1)
        q = K.dot(inputs, self.q_kernel)
        k = K.dot(states[1], self.k_kernel)
        v = K.dot(states[1], self.v_kernel)
        q = tf.expand_dims(q, 1)
        q = self.split_heads_3to5(q, self.heads)
        k = self.split_heads_3to5(k, self.heads)
        v = self.split_heads_3to5(v, self.heads)
        if training:
            mask_length = states[1].get_shape().as_list()[1]
            if mask_length is None:
                mask_length = 1
            mask_extra = np.random.choice([0, 1], [1, mask_length],
                                          p=[0.9, 0.1])

            mask = tf.cast(tf.greater(states[2] + mask_extra, 0), tf.float32)
            mask = tf.cast(mask * -1e9, tf.float32)
        else:
            mask = tf.cast(states[2] * -1e9, tf.float32)
        # mask = tf.cast(states[3] * -1e9, tf.float32)

        mask = tf.expand_dims(mask, 1)
        mask = tf.expand_dims(mask, 1)
        mask = tf.expand_dims(mask, 1)
        if training:
            a, self.attention_weight = hyper_fn.scaled_dot_product_attention(
                q, k, v, mask=mask, dropout=self.dropout)
        else:
            a, self.attention_weight = hyper_fn.scaled_dot_product_attention(
                q, k, v, mask=mask)

        att = self.combine_heads(a)
        self.attention_output = tf.squeeze(att, axis=1)
        # att = tf.transpose(att, [0, 2, 1, 3])
        # att = tf.reshape(att, [batch_size, self.heads, -1])
        return self.attention_output

    def set_attention_context(self, context):
        self.context = context

    def set_external_attention_context(self, context):
        self.external_context = context

    def get_attention_context(self):
        return self.context

    def get_external_attention_context(self):
        return self.external_context

    def get_attention_weights(self):
        return self.attention_weight

    def get_attention_output(self):
        return self.attention_output

    def get_external_attention_weights(self):
        return self.external_attention_weight

    def split_heads_3to5(self, x, heads):
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]
            num_units = tf.shape(x)[-1]
            # Calculate depth of last dimension after it has been split.
            depth = (self.units // heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, 3, num_units // 3])
            x = tf.transpose(x, [0, 2, 1, 3])
            x = tf.reshape(x, [batch_size, 3, length, heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 1, 3, 2, 4])

    def split_heads_2to3(self, x, heads):
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            num_units = tf.shape(x)[-1]
            # Calculate depth of last dimension after it has been split.
            depth = (self.units // heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, 3, num_units // 3])
            x = tf.reshape(x, [batch_size, 3, heads, depth])

            # Transpose the result
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, [batch_size, heads, -1])

    def combine_heads(self, x):
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[3]
            # num_units = tf.shape(x)[-1]
            x = tf.transpose(
                x,
                [0, 1, 3, 2, 4])  # --> [batch,chunk, length, num_heads, depth]
            x = tf.reshape(x, [batch_size, 3, length, self.units])
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, [batch_size, length, self.units * 3])

    def get_config(self):
        config = {
            'units': self.units,
            'dropout': self.dropout,
            'implementation': self.implementation,
            'residual': self.residual,
            'heads': self.heads,
            'att': self.att
        }
        base_config = super(HeadsGRUcell, self).get_config()
        del base_config['cell'], base_config['dropout'], base_config[
            'implementation']
        return dict(list(base_config.items()) + list(config.items()))


class HeadsGRU(tf.keras.layers.RNN):
    def __init__(self,
                 units,
                 dropout=0.,
                 att=True,
                 layers=1,
                 return_sequences=True,
                 return_state=True,
                 go_backwards=False,
                 time_major=True,
                 residual=True,
                 external=False,
                 heads=4,
                 name='LSTM_Att',
                 **kwargs):
        # assert time_major is not True, 'Currently, we dont support time_major'
        cell = []
        for i in range(layers):
            cell.append(
                HeadsGRUcell(units,
                             dropout,
                             att=True,
                             name=name,
                             implementation=2,
                             residual=residual,
                             external=external,
                             heads=heads,
                             **kwargs))
        # cell = hyper_layer.NormBlock(cell, self.dropout)
        self.external = external
        # self.units = units
        # self.dropout = dropout
        # cell = tf.keras.layers.LSTMCell(units,
        #                                 name=name,
        #                                 implementation=2,
        #                                 # residual=residual,
        #                                 **kwargs)
        # self.layer_norm = hyper_layer.LayerNorm()
        self.layers = layers
        self.att = att
        self.time_major = time_major
        self.heads = heads
        self._num_constants = 2
        # self.rnn = tf.keras.layers.RNN(cell,
        #                                return_sequences=return_sequences,
        #                                return_state=return_state,
        #                                go_backwards=go_backwards,
        #                                name=name,
        #                                time_major=time_major,
        #                                **kwargs)
        self.initial_input_wrapper = tf.keras.layers.Lambda(
            lambda x: (x[0], x[1], x[2]), name='initial_state')
        super(HeadsGRU, self).__init__(cell,
                                       return_sequences=return_sequences,
                                       return_state=return_state,
                                       go_backwards=go_backwards,
                                       name=name,
                                       time_major=time_major,
                                       **kwargs)

        self.input_spec = [InputSpec(ndim=3)]

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        # inputs, initial_state, constants = self._standardize_args(inputs,
        #                                                  initial_state,
        #                                                  constants,
        #                                                  2)
        inputs, initial_state, constants = self.initial_input_wrapper(
            (inputs, initial_state, constants))
        if initial_state is not None:

            kwargs['initial_state'] = initial_state
        if constants is not None:
            kwargs['constants'] = constants
        return super(HeadsGRU, self).__call__(inputs, **kwargs)

    def call(
            self,
            inputs,
            initial_state=None,
            mask=None,
            training=False,
            constants=None,
    ):
        inputs, initial_state, constants = self._process_inputs(
            inputs, initial_state, constants)
        # if initial_state[0] is None:
        #     batch_size = tf.shape(inputs)[0]
        #     initial_state = tf.keras.backend.zeros(
        #         (batch_size, self.cell.units))
        #     initial_state = hyper_fn.get_position_encoding(
        #         1, self.cell.units) + initial_state
        # initial_state = self.initial_state_wrapper(initial_state)
        # import pdb
        # pdb.set_trace()
        if len(constants) < 2:
            constants = [constants[0], None]
        if self.time_major:
            inputs = tf.transpose(inputs, [1, 0, 2])
        assert (constants[0] is not None or constants[1] is not None
                ) and self.att, 'atteniont mode, please give context'
        for c in self.cell.cells:
            c.reset_dropout_mask()
            c.reset_recurrent_dropout_mask()
        # self.cell.set_attention_context(context)
        # constants = (context, external_context)
        # inputs = self.layer_norm(inputs)
        if self.layers > 1:
            output, _, last = super(HeadsGRU,
                                    self).call(inputs,
                                               mask=mask,
                                               training=training,
                                               initial_state=initial_state,
                                               constants=constants)
        else:
            output, last = super(HeadsGRU,
                                 self).call(inputs,
                                            mask=mask,
                                            training=training,
                                            initial_state=initial_state,
                                            constants=constants)
        # output, last = self.rnn(inputs,
        #                         mask=mask,
        #                         training=training,
        #                         initial_state=initial_state,
        #                         constants=constants)
        if self.time_major:
            output = tf.transpose(output, [1, 0, 2])
        # if training:
        #     output = tf.nn.dropout(output, self.dropout)
        # output = self.layer_norm(inputs + output)
        # output = inputs + output
        return output, last

    def _standardize_args(self, inputs, initial_state, constants,
                          num_constants):
        """Standardizes `__call__` to a single list of tensor inputs.
      When running a model loaded from a file, the input tensors
      `initial_state` and `constants` can be passed to `RNN.__call__()` as part
      of `inputs` instead of by the dedicated keyword arguments. This method
      makes sure the arguments are separated and that `initial_state` and
      `constants` are lists of tensors (or None).
      Arguments:
        inputs: Tensor or list/tuple of tensors. which may include constants
          and initial states. In that case `num_constant` must be specified.
        initial_state: Tensor or list of tensors or None, initial states.
        constants: Tensor or list of tensors or None, constant tensors.
        num_constants: Expected number of constants (if constants are passed as
          part of the `inputs` list.
      Returns:
        inputs: Single tensor or tuple of tensors.
        initial_state: List of tensors or None.
        constants: List of tensors or None.
      """
        if isinstance(inputs, list):
            # There are several situations here:
            # In the graph mode, __call__ will be only called once. The initial_state
            # and constants could be in inputs (from file loading).
            # In the eager mode, __call__ will be called twice, once during
            # rnn_layer(inputs=input_t, constants=c_t, ...), and second time will be
            # model.fit/train_on_batch/predict with real np data. In the second case,
            # the inputs will contain initial_state and constants as eager tensor.
            #
            # For either case, the real input is the first item in the list, which
            # could be a nested structure itself. Then followed by initial_states, which
            # could be a list of items, or list of list if the initial_state is complex
            # structure, and finally followed by constants which is a flat list.
            assert initial_state is None and constants is None
            if num_constants:
                constants = inputs[-num_constants:]
                inputs = inputs[:-num_constants]
            if len(inputs) > 1:
                initial_state = inputs[1:]
                inputs = inputs[:1]

            if len(inputs) > 1:
                inputs = tuple(inputs)
            else:
                inputs = inputs[0]

        def to_list_or_none(x):
            if x is None or isinstance(x, list):
                return x
            if isinstance(x, tuple):
                return list(x)
            return [x]

        initial_state = to_list_or_none(initial_state)
        constants = to_list_or_none(constants)

        return inputs, initial_state, constants

    # @property
    # def units(self):
    #     return self.cell.units
    #
    # @property
    # def dropout(self):
    #     return self.cell.dropout

    def get_attention_weights(self):
        return self.cell.get_attention_weights()

    def get_attention_output(self):
        return self.cell.get_attention_output()

    def get_config(self):
        config = {
            # 'units': self.units,
            # 'dropout': self.dropout,
            # 'implementation': 2,
            # 'residual': self.residual,
            'heads': self.heads,
            'layer': self.layers
        }
        base_config = super(HeadsGRU, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))


# class LayerNorm_LSTM_Attention(tf.keras.layers.Layer):
#     def __init__(self,
#                  units,
#                  dropout=0.,
#                  go_backwards=False,
#                  time_major=True,
#                  name='LSTM',
#                  external=False,
#                  heads=1,
#                  **kwargs):
#         super(LayerNorm_LSTM_Attention, self).__init__(name=name, **kwargs)
#         self.layer_norm = hyper_layer.LayerNorm(name=name)
#         self.units = units
#         self.dropout = dropout
#         self.time_major = time_major
#         self.heads = heads
#         self.LSTM = LSTM_Attention(
#             units,
#             dropout=dropout,
#             return_sequences=True,
#             return_state=True,
#             go_backwards=False,
#             time_major=True,
#             external=external,
#             heads=heads,
#             name=name,
#         )
#
#     def build(self, input_shape):
#         # Create normalization layer
#         # input_dim = input_shape[-1]
#         # if tf.not_equal(input_dim, self.units):
#         #     self.pre_project = tf.keras.layers.Dense(self.units,
#         #                                         use_bias=False,
#         #                                         name='projection')
#
#         super(LayerNorm_LSTM_Attention, self).build(input_shape)
#
#     def get_config(self):
#         return {
#             "dropout": self.dropout,
#             'units': self.units,
#             'heads': self.heads
#         }
#
#     def call(self,
#              x,
#              context,
#              external_context=None,
#              initial_state=None,
#              training=False):
#         """Calls wrapped layer with same parameters."""
#         # Preprocessing: apply layer normalization
#         y = self.layer_norm(x)
#         # Get layer output
#         outputs, last, state = self.LSTM(y,
#                                          context=context,
#                                          external_context=external_context,
#                                          training=training,
#                                          initial_state=initial_state)
#         # Postprocessing: apply dropout and residual connection
#         # if self.time_major:
#         #     outputs = tf.transpose(outputs, [1, 0, 2])
#         #     last = tf.transpose(last, [1, 0, 2])
#         # if training:
#         #     outputs = tf.nn.dropout(outputs, rate=self.dropout)
#         return (outputs + x), (x[:, -1, :] + last), state
#
#     def get_attention_weights(self):
#         return self.LSTM.get_attention_weights()

# class LayerNorm_cudnnLSTM_Attention(tf.keras.layers.Layer):
#     def __init__(self,
#                  units,
#                  dropout=0.,
#                  go_backwards=False,
#                  time_major=True,
#                  name='LSTM',
#                  external=False,
#                  heads=1,
#                  **kwargs):
#         super(LayerNorm_cudnnLSTM_Attention, self).__init__(name=name, **kwargs)
#         self.layer_norm = hyper_layer.LayerNorm(name=name)
#         self.units = units
#         self.dropout = dropout
#         self.time_major = time_major
#         self.heads = heads
#         self.LSTM =  tf.keras.layer.LSTM(units,dropout=dropout,
#             return_sequences=True,
#             return_state=True,
#             go_backwards=False,
#             time_major=True,
#             name=name,
#         )
#     def build(self, input_shape):
#         # Create normalization layer
#         # input_dim = input_shape[-1]
#         # if tf.not_equal(input_dim, self.units):
#         #     self.pre_project = tf.keras.layers.Dense(self.units,
#         #                                         use_bias=False,
#         #                                         name='projection')
#
#         super(LayerNorm_LSTM_Attention, self).build(input_shape)
#
#     def get_config(self):
#         return {
#             "dropout": self.dropout,
#             'units': self.units,
#             'heads': self.heads
#         }
#
#     def call(self,
#              x,
#              context,
#              external_context=None,
#              initial_state=None,
#              training=False,
#              one_shot=False):
#         """Calls wrapped layer with same parameters."""
#         # Preprocessing: apply layer normalization
#         y = self.layer_norm(x)
#         time_step = tf.shape(y)[1]
#         # Get layer output
#         output_grapper =  tf.TensorArray(tf.int64,
#                                                    self.max_seq_length + 1,
#                                                    dynamic_size=True)
#         if one_shot and tf.equal(time_step,1):
#             outputs, last, state = self.LSTM(y,training=training,
#                                              initial_state=initial_state)
#             return outputs, last, state
#         # Postprocessing: apply dropout and residual connection
#         # if self.time_major:
#         #     outputs = tf.transpose(outputs, [1, 0, 2])
#         #     last = tf.transpose(last, [1, 0, 2])
#         # if training:
#         #     outputs = tf.nn.dropout(outputs, rate=self.dropout)
#         return (outputs + x), (x[:, -1, :] + last), state
#
#     def get_attention_weights(self):
#         return self.LSTM.get_attention_weights()

#
# import numpy as np
# data = tf.constant(range(15000), shape=(10, 30, 50), dtype=tf.float32)
# state = tf.ones((10, 20, 50))
# cell = LayerNorm_LSTM_Attention(50, 0.1)
# outputs, state = cell(data, context=state, training=False)
# print(np.shape(outputs))
# # print(np.shape(hidden))
# print(np.shape(state))
# n = 10000
# x = tf.constant(list(range(n)))
# def add(x):
#     return x+1
# c = lambda i, x: i < n
# b = lambda i, x: (i + 1, add(x))
# i, out = tf.while_loop(c, b, (0, x))
# import pdb; pdb.set_trace()
# print(i)
# print(out)

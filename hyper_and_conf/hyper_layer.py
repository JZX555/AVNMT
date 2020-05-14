# encoder=utf8
from hyper_and_conf import hyper_util
import tensorflow as tf
from tensorflow.python.keras import regularizers
from hyper_and_conf import hyper_fn
L2_WEIGHT_DECAY = 1e-6
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


class Pixel_Self_Att(tf.keras.layers.Layer):
    def __init__(self, num_units, num_heads, dropout):
        """
        This only jonitly attend to internal information with in 2D. If you want to
        attend to 3D information, using Atteniton instead.
    Args:
      num_units: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      attention_dropout: float, dropout rate inside attention for training.
    """
        if num_units % num_heads:
            raise ValueError(
                "Hidden size ({}) must be divisible by the number of heads ({})."
                .format(num_units, num_heads))

        super(Pixel_Self_Att, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout = dropout

    def build(self, input_shape):
        """Builds the layer."""
        depth = self.num_units // self.num_heads
        self.time_output_dense_layer = tf.keras.layers.Dense(
            self.num_units, use_bias=False, name="output_transform")
        self.q_dense_layer = tf.keras.layers.Dense(self.num_units,
                                                   use_bias=False,
                                                   name="q")
        self.k_dense_layer = tf.keras.layers.Dense(self.num_units,
                                                   use_bias=False,
                                                   name="k")
        self.v_dense_layer = tf.keras.layers.Dense(self.num_units,
                                                   use_bias=False,
                                                   name="v")
        self.output_dense_layer = tf.keras.layers.Dense(
            self.num_units, use_bias=False, name="output_transform")
        super(Pixel_Self_Att, self).build(input_shape)

    def get_config(self):
        return {
            "num_units": self.num_units,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
        }

    def split_heads(self, x, length_prior=True):
        """Split x into different heads, and transpose the resulting value.
    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.
    Args:
      x: A tensor with shape [batch_size, length, num_units]
    Returns:
      A tensor with shape [batch_size, num_heads, length, num_units/num_heads]
    """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]
            img_size = tf.shape(x)[2]

            # Calculate depth of last dimension after it has been split.
            depth = (self.num_units // self.num_heads)

            # Split the last dimension
            x = tf.reshape(
                x, [batch_size, length, img_size, self.num_heads, depth])

            # Transpose the result
            if length_prior:
                return tf.transpose(x, [0, 3, 2, 1, 4])
            else:
                return tf.transpose(x, [0, 3, 1, 2, 4])

    def combine_heads(self, x, length_prior=True):
        """Combine tensor that has been split.
    Args:
      x: A tensor [batch_size, num_heads, length, num_units/num_heads]
    Returns:
      A tensor with shape [batch_size, length, num_units]
    """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            if length_prior:
                length = tf.shape(x)[-2]
                img_size = tf.shape(x)[-3]
                x = tf.transpose(
                    x,
                    [0, 3, 2, 1, 4])  # --> [batch, length, num_heads, depth]
            else:
                length = tf.shape(x)[-3]
                img_size = tf.shape(x)[-2]

                x = tf.transpose(x, [0, 2, 3, 1, 4])
            return tf.reshape(x, [batch_size, length, -1, self.num_units])

    def call(self, x, bias, training, cache=None):
        """Apply attention mechanism to x and y.
    Args:
      x: a tensor with shape [batch_size, length_x, num_units]
      y: a tensor with shape [batch_size, length_y, num_units]
      bias: attention bias that will be added to the result of the dot product.
      training: boolean, whether in training mode or not.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.
    Returns:
      Attention layer output with shape [batch_size, length_x, num_units]
    """
        padding_bias = tf.cast(tf.not_equal(x, 0), tf.float32)
        q_org = self.q_dense_layer(x)
        k_org = self.k_dense_layer(x)
        v_org = self.v_dense_layer(x)
        q = self.split_heads(q_org, length_prior=False)
        k = self.split_heads(k_org, length_prior=False)
        v = self.split_heads(v_org, length_prior=False)
        attention_output, _ = hyper_fn.scaled_dot_product_attention(q, k, v)

        length_attention_output = self.combine_heads(attention_output,
                                                     length_prior=False)

        return self.time_output_dense_layer(
            length_attention_output) * padding_bias


class Attention(tf.keras.layers.Layer):
    """Multi-headed attention layer."""
    def __init__(self, num_units, num_heads, dropout):
        """Initialize Attention.
    Args:
      num_units: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      attention_dropout: float, dropout rate inside attention for training.
    """
        if num_units % num_heads:
            raise ValueError(
                "Hidden size ({}) must be divisible by the number of heads ({})."
                .format(num_units, num_heads))

        super(Attention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_weights = 0

    def build(self, input_shape):
        """Builds the layer."""
        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = tf.keras.layers.Dense(self.num_units,
                                                   use_bias=False,
                                                   name="q")
        self.k_dense_layer = tf.keras.layers.Dense(self.num_units,
                                                   use_bias=False,
                                                   name="k")
        self.v_dense_layer = tf.keras.layers.Dense(self.num_units,
                                                   use_bias=False,
                                                   name="v")
        self.output_dense_layer = tf.keras.layers.Dense(
            self.num_units, use_bias=False, name="output_transform")

        super(Attention, self).build(input_shape)

    def get_config(self):
        return {
            "num_units": self.num_units,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
        }

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.
    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.
    Args:
      x: A tensor with shape [batch_size, length, num_units]
    Returns:
      A tensor with shape [batch_size, num_heads, length, num_units/num_heads]
    """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.num_units // self.num_heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.
    Args:
      x: A tensor [batch_size, num_heads, length, num_units/num_heads]
    Returns:
      A tensor with shape [batch_size, length, num_units]
    """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(
                x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return tf.reshape(x, [batch_size, length, self.num_units])

    def call(self, x, y, bias, training, cache=None, **kwargs):
        """Apply attention mechanism to x and y.
    Args:
      x: a tensor with shape [batch_size, length_x, num_units]
      y: a tensor with shape [batch_size, length_y, num_units]
      bias: attention bias that will be added to the result of the dot product.
      training: boolean, whether in training mode or not.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.
    Returns:
      Attention layer output with shape [batch_size, length_x, num_units]
    """
        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).
        padding_bias = tf.expand_dims(
            tf.cast(tf.not_equal(tf.reduce_sum(x, -1), 0), tf.float32), -1)
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat((cache["k"], k), axis=2)
            v = tf.concat((cache["v"], v), axis=2)

            # Update cache
            cache["k"] = k
            cache["v"] = v
        # Scale q to prevent the dot product between q and k from growing too large.
        # depth = (self.num_units // self.num_heads)
        # q *= depth**-0.5
        # # Calculate dot product attention
        # logits = tf.matmul(q, k, transpose_b=True)
        # logits += bias
        # weights = tf.nn.softmax(logits, name="attention_weights")
        # if training:
        #     weights = tf.nn.dropout(weights, rate=self.dropout)
        # with tf.name_scope('attention_output'):
        #     attention_output = tf.matmul(weights, v)
        if training:
            attention_output, self.attention_weights = hyper_fn.scaled_dot_product_attention(
                q, k, v, mask=bias, dropout=self.dropout)
        else:
            attention_output, self.attention_weights = hyper_fn.scaled_dot_product_attention(
                q, k, v, mask=bias)
        # Recombine heads --> [batch_size, length, num_units]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.output_dense_layer(
            attention_output) * padding_bias
        return attention_output

    def get_attention_weights(self):
        return self.attention_weights


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""
    def call(self, x, bias, training, cache=None, **kwargs):
        return super(SelfAttention, self).call(x, x, bias, training, cache,
                                               **kwargs)


# @tf.keras.utils.register_keras_serializable(package="Text")
class EmbeddingSharedWeights(tf.keras.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""
    def __init__(self, vocab_size, num_units, pad_id, name="embedding"):
        """Specify characteristic parameters of embedding layer.
    Args:
      vocab_size: Number of tokens in the embedding. (Typically ~32,000)
      num_units: Dimensionality of the embedding. (Typically 512 or 1024)
      method: Strategy for performing embedding lookup. "gather" uses tf.gather
        which performs well on CPUs and GPUs, but very poorly on TPUs. "matmul"
        one-hot encodes the indicies and formulates the embedding as a sparse
        matrix multiplication. The matmul formulation is wasteful as it does
        extra work, however matrix multiplication is very fast on TPUs which
        makes "matmul" considerably faster than "gather" on TPUs.
    """
        super(EmbeddingSharedWeights, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.pad_id = pad_id
        self.shared_weights = self.add_weight(
            shape=[self.vocab_size, self.num_units],
            dtype="float32",
            name="shared_weights",
            initializer="glorot_uniform")
        # self.porjection = self.add_weight(
        #     shape=[self.vocab_size, self.num_units],
        #     dtype="float32",
        #     name="project",
        #     initializer=tf.random_normal_initializer(
        #         mean=0., stddev=self.num_units**-0.5))

    def build(self, input_shape):
        super(EmbeddingSharedWeights, self).build(input_shape)
        # self.build = True

    def call(self, inputs, linear=False):
        if linear:
            return self._linear(inputs)
        else:
            return self._embedding(inputs)

    def _embedding(self, inputs):
        embeddings = tf.gather(self.shared_weights, inputs)
        mask = tf.cast(tf.not_equal(inputs, 0), embeddings.dtype)
        embeddings *= tf.expand_dims(mask, -1)
        # Scale embedding by the sqrt of the hidden size
        embeddings *= self.num_units**0.5
        # embeddings = tf.nn.tanh(embeddings)
        return embeddings

    def _linear(self, inputs):
        """Computes logits by running x through a linear layer.
    Args:
      x: A float32 tensor with shape [batch_size, length, num_units]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
        batch_size = tf.shape(input=inputs)[0]
        length = tf.shape(input=inputs)[1]
        inputs = tf.reshape(inputs, [-1, self.num_units])
        logits = tf.matmul(inputs, self.shared_weights, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, self.vocab_size])

    def product(self, inputs):
        batch_size = tf.shape(input=inputs)[0]
        length = tf.shape(input=inputs)[1]
        inputs = tf.reshape(inputs, [-1, self.vocab_size])
        logits = tf.matmul(inputs, self.shared_weights)

        return tf.reshape(logits, [batch_size, length, self.num_units])

    def att_shared_weights(self, inputs):
        # projection = tf.matmul(
        #     self.project_weights, self.shared_weights, transpose_b=True)
        batch_size = tf.shape(input=inputs)[0]
        inputs = tf.reshape(inputs, [-1, self.num_units]) * 64**-0.5
        weights = tf.matmul(inputs, self.shared_weights, transpose_b=True)
        # weights = tf.nn.softmax(weights, -1)
        att = tf.matmul(weights, self.shared_weights)
        att = tf.reshape(inputs, [batch_size, -1, self.num_units])
        return att

    def get_config(self):
        # config = super(EmbeddingSharedWeights, self).get_config()
        c = {
            'vocab_size': self.vocab_size,
            'num_units': self.num_units,
            'pad_id': self.pad_id
        }
        # config.update(c)
        return c


class LayerNorm(tf.keras.layers.Layer):
    """
        Layer normalization for transformer, we do that:
            ln(x) = α * (x - μ) / (σ**2 + ϵ)**0.5 + β
        mode:
            add: ln(x) + x
            norm: ln(x)
    """
    def __init__(self,
                 epsilon=1e-6,
                 gamma_initializer="ones",
                 beta_initializer="zeros",
                 model="linear",
                 name='norm'):
        super(LayerNorm, self).__init__(name=name)
        self.epsilon = epsilon
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.model = model

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.gamma_kernel = self.add_weight(shape=(input_dim),
                                            name="gamma",
                                            initializer=self.gamma_initializer)
        self.beta_kernel = self.add_weight(shape=(input_dim),
                                           name="beta",
                                           initializer=self.beta_initializer)
        super(LayerNorm, self).build(input_shape)

    def call(self, inputs, training=False):
        # inputs = self.mask(inputs)
        # bias = hyper_util.zero_masking(inputs)
        if self.model == "tanh":
            mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
            tanh_estimator = tf.cast(0.01 * ((inputs-mean)/(variance+self.epsilon)),tf.float32)
            normalized = 0.5 * (tf.nn.tanh(tanh_estimator) + 1.0)
        else:
            mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
            normalized = (inputs - mean) * tf.math.rsqrt(variance +
                                                         self.epsilon)
        output = self.gamma_kernel * normalized + self.beta_kernel
        return output

    def get_config(self):
        # config = super(LayerNorm, self).get_config()
        c = {'epsilon': self.epsilon}
        # config.update(c)
        return c


class NormBlock(tf.keras.layers.Layer):
    """Wrapper class that applies layer pre-processing and post-processing."""
    def __init__(self, layer, dropout, add_mode=True):
        super(NormBlock, self).__init__()
        self.layer = layer
        # self.num_units = num_units
        self.dropout = dropout
        self.add_mode = add_mode

    def build(self, input_shape):
        # Create normalization layer
        self.layer_norm = LayerNorm()
        super(NormBlock, self).build(input_shape)

    def get_config(self):
        return {"dropout": self.dropout, 'add_mode': self.add_mode}

    def call(self, x, *args, **kwargs):
        """Calls wrapped layer with same parameters."""
        # Preprocessing: apply layer normalization
        training = kwargs["training"]
        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)
        # if isinstance(y,tuple):
        #     y = y[0]
        # Postprocessing: apply dropout and residual connection
        if training:
            y = tf.nn.dropout(y, rate=self.dropout)
        return y + x


class CNN_FNN(tf.keras.layers.Layer):
    def __init__(self, num_units, dropout, kernel=1, strides=1):
        """Initialize FeedForwardNetwork.
        Args:
          num_units: int, output dim of hidden layer.
          filter_size: int, filter size for the inner (first) dense layer.
          relu_dropout: float, dropout rate for training.
        """
        super(CNN_FNN, self).__init__()
        self.num_units = num_units
        self.kernel = kernel
        self.strides = 1
        self.dropout = dropout

    def build(self, input_shape):
        out_dim = input_shape[-1]
        self.cnn_ffn = ResnetBlock2D(
            self.kernel,
            [out_dim, out_dim * 4, out_dim],
            stage='cnn_ffn',
            block='cnn_fnn',
        )
        super(CNN_FNN, self).build(input_shape)

    def call(self, inputs):
        return self.cnn_ffn(inputs)

    def get_config(self):
        return {
            "kernel": self.kernel,
            "strides": self.strides,
            "num_units": self.num_units,
            "dropout": self.dropout,
        }


class Feed_Forward_Network(tf.keras.layers.Layer):
    """Fully connected feedforward network."""
    def __init__(self,
                 num_units,
                 dropout,
                 activation_filter='relu',
                 activation_output=None):
        """Initialize FeedForwardNetwork.
    Args:
      num_units: int, output dim of hidden layer.
      filter_size: int, filter size for the inner (first) dense layer.
      relu_dropout: float, dropout rate for training.
    """
        super(Feed_Forward_Network, self).__init__()
        self.num_units = num_units
        self.dropout = dropout
        self.activation_filter = activation_filter
        self.activation_output = activation_output

    def build(self, input_shape):
        out_dim = input_shape[-1]
        self.filter_dense_layer = tf.keras.layers.Dense(
            self.num_units,
            use_bias=True,
            activation=self.activation_filter,
            name="filter_layer")
        self.output_dense_layer = tf.keras.layers.Dense(
            out_dim,
            use_bias=True,
            activation=self.activation_output,
            name="output_layer")
        # self.mask = tf.keras.layers.Masking(0)
        super(Feed_Forward_Network, self).build(input_shape)

    def get_config(self):
        return {
            "num_units": self.num_units,
            "dropout": self.dropout,
        }

    def call(self, x, training):
        """Return outputs of the feedforward network.
    Args:
      x: tensor with shape [batch_size, length, num_units]
      training: boolean, whether in training mode or not.
    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, num_units]
    """
        # Retrieve dynamically known shapes
        # batch_size = tf.shape(x)[0]
        # length = tf.shape(x)[1]
        output = self.filter_dense_layer(x)
        if training:
            output = tf.nn.dropout(output, rate=self.dropout)
        output = self.output_dense_layer(output)

        return output


class SequenceResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_units, dropout):
        self.dropout = dropout
        self.filters = num_units
        super(SequenceResnetBlock, self).__init__('res_block')

    def build(self, input_shape):
        # self.filters = tf.shape(input_shape)[-1]
        self.bottel_fillters = int(self.filters / 4)
        self.shortcut_projection = tf.keras.layers.Conv1D(
            1,
            1,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))
        self.pre_norm = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
        )

        self.conv1a = tf.keras.layers.Conv1D(
            self.bottel_fillters,
            1,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))
        self.bn1a = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
        )

        self.conv1b = tf.keras.layers.Conv1D(
            self.bottel_fillters,
            3,
            padding='SAME',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))
        self.bn1b = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
        )

        self.conv1c = tf.keras.layers.Conv1D(
            self.filters,
            1,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        )
        self.bn1c = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
        )
        super(SequenceResnetBlock, self).build(input_shape)

    def get_config(self):
        return {"dropout": self.dropout, 'filters': self.filters}

    def call(self, inputs, post_norm=True, training=False):
        with tf.name_scope('res_block'):
            # shortcut = self.shortcut_projection(inputs)Vj
            # input_tensor = self.pre_norm(inputs)
            # input_tensor = tf.nn.relu(input_tensor)
            shortcut = inputs
            x = self.conv1a(inputs)
            x = self.bn1a(x, training=training)
            x = tf.nn.relu(x)

            x = self.conv1b(x)
            x = self.bn1b(x, training=training)
            x = tf.nn.relu(x)

            x = self.conv1c(x)
            if post_norm:
                x = self.bn1c(x, training=training)
                # x = tf.keras.layers.BatchNormalization()(x)
                if training:
                    x = tf.nn.dropout(x, rate=self.dropout)
                x = x + shortcut
                return tf.nn.relu(x)
            else:
                if training:
                    x = tf.nn.dropout(x, rate=self.dropout)
                x = x + shortcut
                return x


class StackedSeqResBlock(tf.keras.layers.Layer):
    def __init__(self, num_units, dropout, layers):
        self.num_units = num_units
        self.dropout = dropout
        self.layers = layers
        super(StackedSeqResBlock, self).__init__(name='StackedSeqResBlock')

    def get_config(self):
        return {'dropout': self.dropout, 'layer': self.layers}

    def build(self, input_shape):
        self.stacked_block = []
        for i in range(self.layers):
            self.stacked_block.append(
                SequenceResnetBlock(self.num_units, self.dropout))
        super(StackedSeqResBlock, self).build(input_shape)

    def call(self, inputs, post_norm=True, training=False):
        for index, layer in enumerate(self.stacked_block):
            with tf.name_scope('res_layer_%d' % index):
                inputs = layer(inputs, post_norm=post_norm, training=training)

        return inputs


class ResnetBlock2D(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size,
                 filters,
                 strides=(1, 1),
                 stage=0,
                 block=0,
                 dropout=0.1,
                 mode='identity'):
        super(ResnetBlock2D, self).__init__()
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = -1
        self.dropout = dropout
        self.mode = mode
        self.conv2a = tf.keras.layers.Conv2D(
            filters1,
            (1, 1),
            use_bias=False,
            kernel_initializer='he_normal',
            # kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + '2a')
        self.bn2a = tf.keras.layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=bn_name_base + '2a')
        # self.bn2a = LayerNorm(name=bn_name_base + '2a')

        self.conv2b = tf.keras.layers.Conv2D(
            filters2,
            kernel_size,
            strides=strides,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            # kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + '2b')
        self.bn2b = tf.keras.layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=bn_name_base + '2b')
        self.bn2b = LayerNorm(name=bn_name_base + '2b')

        self.conv2c = tf.keras.layers.Conv2D(
            filters3,
            (1, 1),
            use_bias=False,
            kernel_initializer='he_normal',
            # kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + '2c')
        self.bn2c = tf.keras.layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=bn_name_base + '2c')
        # self.bn2c = LayerNorm(name=bn_name_base + '2c')
        if mode == 'conv':
            self.short_cut = tf.keras.layers.Conv2D(
                filters3,
                (1, 1),
                strides=strides,
                use_bias=False,
                kernel_initializer='he_normal',
                # kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                name=conv_name_base + '1')
            self.short_cut_norm = tf.keras.layers.BatchNormalization(
                axis=bn_axis,
                momentum=BATCH_NORM_DECAY,
                epsilon=BATCH_NORM_EPSILON,
                name=bn_name_base + '1')
            # self.short_cut_norm = LayerNorm(name=bn_name_base + '1')

    def call(self, input_tensor, padding=1, training=False):
        x = self.conv2a(input_tensor)
        padding = tf.cast(tf.not_equal(x, 0), tf.float32)
        x = self.bn2a(x, training=training) * padding
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        padding = tf.cast(tf.not_equal(x, 0), tf.float32)
        x = self.bn2b(x, training=training) * padding
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        padding = tf.cast(tf.not_equal(x, 0), tf.float32)
        x = self.bn2c(x, training=training) * padding
        if self.mode == 'conv':
            input_tensor = self.short_cut(input_tensor)
            input_tensor = self.short_cut_norm(input_tensor) * padding
        # if training:
        #     dropout = tf.nn.dropout(x, self.dropout)
        x += input_tensor
        return tf.nn.relu(x)


class Headed_Feed_Forward_Network(tf.keras.layers.Layer):
    """Fully connected feedforward network."""
    def __init__(self, heads, dropout):
        """Initialize FeedForwardNetwork.
    Args:
      num_units: int, output dim of hidden layer.
      filter_size: int, filter size for the inner (first) dense layer.
      relu_dropout: float, dropout rate for training.
    """
        super(Headed_Feed_Forward_Network, self).__init__()
        self.heads = heads
        self.dropout = dropout

    def build(self, input_shape):
        out_dim = input_shape[-1]
        self.depth = out_dim // self.heads
        self.filter_dense_layer = tf.keras.layers.Dense(out_dim,
                                                        use_bias=True,
                                                        activation=tf.nn.relu,
                                                        name="filter_layer")
        self.output_dense_layer = tf.keras.layers.Dense(out_dim,
                                                        use_bias=True,
                                                        name="output_layer")
        # self.mask = tf.keras.layers.Masking(0)
        super(Headed_Feed_Forward_Network, self).build(input_shape)

    def get_config(self):
        return {
            "heads": self.heads,
            "dropout": self.dropout,
        }

    def call(self, x, training):
        """Return outputs of the feedforward network.
    Args:
      x: tensor with shape [batch_size, length, num_units]
      training: boolean, whether in training mode or not.
    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, num_units]
    """
        # Retrieve dynamically known shapes
        batch_size = tf.shape(x)[0]
        # length = tf.shape(x)[1]
        output = self.filter_dense_layer(x)
        output = tf.reshape(output, [batch_size, self.heads, self.depth])
        if training:
            output = tf.nn.dropout(output, rate=self.dropout)
        output = self.output_dense_layer(output)

        return output


# class ResnetIdentityBlock(tf.keras.layers.Layer):
#     def __init__(self, kernel_size, filters):
#         self.filters1, self.filters2, self.filters3 = filters
#         self.kernel_size = kernel_size
#         self.filters = filters
#         super(ResnetIdentityBlock, self).__init__('res_block')
#
#     def build(self, input_shape):
#         self.conv1a = tf.keras.layers.Conv1D(self.filters1, 1)
#         self.bn1a = tf.keras.layers.BatchNormalization()
#
#         self.conv1b = tf.keras.layers.Conv1D(
#             self.filters2, self.kernel_size, padding='same')
#         self.bn1b = tf.keras.layers.BatchNormalization()
#
#         self.conv1c = tf.keras.layers.Conv1D(self.filters3, 1)
#         self.bn1c = tf.keras.layers.BatchNormalization()
#         super(ResnetIdentityBlock, self).build(input_shape)
#
#     def get_config(self):
#         return {"kernel_size": self.kernel_size, 'filters': self.filters}
#
#     def call(self, input_tensor, training=False):
#         x = self.conv1a(input_tensor)
#         x = self.bn1a(x, training=training)
#         x = tf.nn.relu(x)
#
#         x = self.conv1b(x)
#         x = self.bn1b(x, training=training)
#         x = tf.nn.relu(x)
#
#         x = self.conv1c(x)
#         x = self.bn1c(x, training=training)
#
#         x += input_tensor
#         return tf.nn.relu(x)

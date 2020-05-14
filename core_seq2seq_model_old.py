import numpy as np
import tensorflow as tf
import core_heads_gru
import core_gru
import model_helper
import random
from hyper_and_conf import hyper_layer, hyper_beam_search

class Attention(tf.keras.layers.Layer):
    def __init__(self, num_units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(num_units)
        self.W2 = tf.keras.layers.Dense(num_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        if len(np.shape(query)) == 2:
            query_with_time_axis = tf.expand_dims(query, 1)
        else:
            query_with_time_axis = query

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_units,
                 batch_size,
                 dec_hidden_size,
                 drop_out = 1,
                 kernel_initializer = 'glorot_uniform',
                 name = None,
                 **kwargs):
        super(Encoder, self).__init__(name = name, **kwargs)

        self.num_units = num_units
        self.batch_size = batch_size
        self.dec_hidden_size = dec_hidden_size
        self.drop_out = max(0, min(drop_out, 1))

        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.forward_cell = core_gru.GRU(
            units = self.num_units,
            dropout = self.drop_out,
            return_sequences = True,
            return_state = True,
            name = 'forward_Encoder')

        self.backward_cell = core_gru.GRU(
            units = self.num_units,
            dropout = self.drop_out,
            return_sequences = True,
            return_state = True,
            go_backwards=True,
            name = 'backward_Eecoder')
        
        self.bi_cell = tf.keras.layers.Bidirectional(self.forward_cell, 
                                                    backward_layer=self.backward_cell,)
        
        # kernel_shape = (self.num_units, self.dec_hidden_size)
        # self.fc_kernel = self.add_weight(
        #     shape = kernel_shape,
        #     name = 'encoder_fc',
        #     initializer=self.kernel_initializer
        # )

    def call(self, inputs, init_hidden = None, training = False):
        if 0 < self.drop_out < 1:
            self.dropout_mask = model_helper.dropout_mask_helper(
                tf.ones_like(inputs), self.drop_out, training = False)

            inputs = inputs * self.dropout_mask 
        
        if not init_hidden:
            init_hidden = self.get_init_hidden(tf.shape(inputs)[0])
            init_hidden = [init_hidden, init_hidden]

        outputs, f_hidden, b_hidden= self.bi_cell(inputs, initial_state = init_hidden, training = training)

        # hidden = tf.concat(hidden, axis = -1)
        # hidden = tf.nn.tanh(tf.keras.backend.dot(hidden, self.fc_kernel))

        return outputs, b_hidden
    
    def get_init_hidden(self, batch_size):
        return tf.zeros((batch_size, self.num_units))

class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_units,
                 embedded_size,
                 output_size,
                 drop_out = 1,
                 kernel_initializer = 'glorot_uniform',
                 name = None,
                 **kwargs):
        super(Decoder, self).__init__(name = name, **kwargs)

        self.num_units = num_units
        self.embedded_size = embedded_size
        self.output_size = output_size

        self.drop_out = max(0, min(drop_out, 1))
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.cell = core_gru.GRU(
            units = self.num_units,
            dropout = self.drop_out,
            return_sequences = True,
            return_state = True,
            name = 'Decoder')

        self.fc_shape = (self.num_units, self.output_size)
        self.fc_kernel = self.add_weight(
            shape = self.fc_shape,
            name = 'decoder_fc',
            initializer = self.kernel_initializer
        )

        self.attention = Attention(self.num_units)
    
    def call(self, 
             inputs, 
             initial_state, 
             context, 
             training=False):
        if 0 < self.drop_out < 1:
            self.dropout_mask = model_helper.dropout_mask_helper(
                tf.ones_like(inputs), self.drop_out, training=True)
            self.att_dropout_mask = model_helper.dropout_mask_helper(
                tf.ones_like(context), self.drop_out, training=True)

            inputs = inputs * self.dropout_mask 
            context = context * self.att_dropout_mask

        att_inputs, att_weight = self.attention(inputs, context)

        inputs = tf.concat((inputs, tf.expand_dims(att_inputs, axis = 1)), axis = -1)

        output, hidden = self.cell(  
                    inputs,
                    initial_state=initial_state,
                    training=training,
                )

        # att = tf.expand_dims(self.cell.get_attention_output(), axis = 1)
        
        # output = tf.concat([output, inputs, att], axis = -1)
        output = tf.keras.backend.dot(output, self.fc_kernel)

        return output, hidden

class Seq2Seq(tf.keras.layers.Layer):
    def __init__(self,
                 vocabulary,
                 enc_units,
                 dec_units,
                 batch_size,
                 embedded_size,
                 output_size,
                 drop_out = 1,
                 max_seq_len = 50,
                 SOS_ID = 1,
                 EOS_ID = 2,
                 PAD_ID = 0,
                 MASK_ID = 3,
                 UNK_ID = 4,
                 name = None,
                 **kwargs):
        super(Seq2Seq, self).__init__(name = name, **kwargs)
        self.vocabulary = vocabulary
        self.enc_units = enc_units
        self.dec_units = dec_units
        self.batch_size = batch_size
        self.embedded_size = embedded_size
        self.output_size = output_size

        self.drop_out = max(0, min(drop_out, 1))
        self.max_seq_len = max_seq_len

        self.EOS_ID = EOS_ID
        self.PAD_ID = PAD_ID
        self.SOS_ID = SOS_ID
        self.UNK_ID = UNK_ID
        self.MASK_ID = MASK_ID

    
    def build(self, input_shape):
        self.embedding_softmax_layer = hyper_layer.EmbeddingSharedWeights(
            self.vocabulary, self.embedded_size, pad_id=self.PAD_ID)

        self.encoder = Encoder(self.enc_units, 
                        self.batch_size,
                        self.dec_units, 
                        drop_out = self.drop_out,
                    )
        self.decoder = Decoder(self.dec_units, 
                        self.embedded_size, 
                        self.output_size,
                        drop_out = self.drop_out,
                    )

    def call(self, 
             inputs, 
             teacher_force_rate = 0.85, 
             inference = False,
             use_while_loop = True):
        (src, tgt) = inputs
        
        mask = tf.math.logical_not(tf.math.equal(tgt, self.PAD_ID))
        tgt = tf.pad(tgt, [[0, 0], [1, 0]],
                        constant_values=self.SOS_ID)

        src = self.embedding_softmax_layer(src)
        tgt = self.embedding_softmax_layer(tgt)

        tgt_seq_len = tgt.shape[1]

        enc_output, hidden = self.encoder(src)

        outputs = []
        output = tgt[:, 0, :]
        output = tf.expand_dims(output, axis = 1)

        if use_while_loop:
            i = 1
            cond = lambda i, tgt, inputs, enc_output, hidden, outputs, teacher_force_rate: i < tgt_seq_len
            tf.while_loop(cond, self.loop_body, loop_vars = [i, tgt, output, enc_output, hidden, outputs, teacher_force_rate])

        else:
            for i in range(1, tgt_seq_len):
                dec_output, hidden = self.decoder(output, 
                                    initial_state = hidden, 
                                    context = enc_output,
                                )
                outputs.append(dec_output)

                teacher_force = random.random() < teacher_force_rate     
                if teacher_force is not None:
                    output = tgt[:, i, :]
                    output = tf.expand_dims(output, axis = 1)
                else:
                    output = self.embedding_softmax_layer(tf.argmax(output, axis=-1))

        logits = tf.concat(outputs, axis = 1)
        # logits = tf.transpose(logits, [1, 0, 2])

        mask = tf.cast(mask, dtype=logits.dtype)
        mask = tf.tile(tf.expand_dims(mask, axis = -1), [1, 1, self.output_size])
        logits *= mask

        return logits

    def inference(self, inputs):
        src = self.embedding_softmax_layer(inputs)

        enc_output, hidden = self.encoder(src)

        outputs = self.predict(initial_state=hidden, context = enc_output)

        return outputs

    def get_predict_with_logits(self, logits):
        return tf.argmax(logits, axis=-1, output_type=tf.int32)

    def loop_body(self, i, tgt, inputs, enc_output, hidden, outputs, teacher_force_rate):
        dec_output, hidden = self.decoder(inputs, 
                        initial_state = hidden, 
                        context = enc_output,
                    )
        outputs.append(dec_output)

        teacher_force = random.random() < teacher_force_rate     
        if teacher_force is not None:
            inputs = tgt[:, i, :]
            inputs = tf.expand_dims(inputs, axis = 1)
        else:
            inputs = self.embedding_softmax_layer(tf.argmax(inputs, axis=-1)) 

        return i + 1, tgt, inputs, enc_output, hidden, outputs, teacher_force_rate

    def predict(self,
                initial_state=None,
                context=None,
                training=False):
        def _loop_fn(inputs, i, cache):
            inputs = tf.expand_dims(inputs[:, i], -1)
            inputs = self.embedding_softmax_layer(inputs)

            logits, last = self.decoder(
                inputs,
                context=cache['context'],
                initial_state=cache['initial_state'],
                training=training)

            cache['initial_state'] = last
            # logits = self.embedding_softmax_layer(logits, linear=True)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        batch_size = tf.shape(context)[0]
        max_decode_length = self.max_seq_len
        initial_ids = tf.zeros([batch_size], dtype=tf.int32) + self.SOS_ID
        # initial_c = tf.keras.backend.zeros_like(initial_state)
        # initial_state = (initial_state, initial_c)

        cache = {
            'context': context,
            'initial_state': initial_state,
        }
        beam_size = 4
        alpha = 0.6
        end = self.EOS_ID
        decoded_ids, scores = hyper_beam_search.sequence_beam_search(
            symbols_to_logits_fn=_loop_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.vocabulary,
            beam_size=beam_size,
            alpha=alpha,
            max_decode_length=max_decode_length,
            eos_id=end)
        # Get the top sequence for each batch element
        top_decoded_ids = decoded_ids[:, 0, 1:]
        # top_scores = scores[:, 0]
        return top_decoded_ids
    
    def get_loss(self, labels, logits):
        logits = tf.reshape(logits, [-1, self.output_size])
        labels = tf.reshape(labels, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        mask = tf.math.logical_not(tf.math.equal(labels, 0))
        mask = tf.cast(mask, dtype=loss.dtype)

        return tf.reduce_mean(loss * mask)


if __name__ == '__main__':
    # import time

    # src = tf.constant(range(100), shape = [5, 20], dtype = tf.int32)
    # tgt = tf.constant(range(100), shape = [5, 20], dtype = tf.int32)

    # inputs = (src, tgt)
    # model = Seq2Seq(vocabulary = 1000,
    #                 enc_units = 160,
    #                 dec_units = 160,
    #                 batch_size = 5,
    #                 embedded_size = 160,
    #                 output_size = 1000,)

    # begin = time.time()
    # logits = model(inputs)
    # end = time.time()
    # print(end - begin)
    # print('loss is: {}'.format(model.get_loss(tgt, logits)))
    # predict = model.get_predict_with_logits(logits)
    # # print(logits)
    # print(predict)

    # ids = model.inference(src)
    # print(ids)
    src = tf.constant(range(100), shape = [2, 5, 10], dtype = tf.float32)

    model = Encoder(10, 2, 10)

    o, h = model(src)

    print(o)
    print(h)
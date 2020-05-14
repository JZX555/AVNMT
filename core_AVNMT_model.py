import tensorflow as tf
import numpy as np
import random

import core_seq2seq_model
from tensorflow.python.keras import backend as K
from hyper_and_conf import hyper_layer, hyper_beam_search

class VAE(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 name='VAE',
                 **kwargs,
                 ):
        super(VAE, self).__init__(name=name, **kwargs)

        self.units = units

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
    
    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.fc_kernel = self.add_weight(
            shape = (input_dim, self.units),
            initializer = self.kernel_initializer,
            name = 'VAE_fc_kernel',            
        )

        self.mu_kernel = self.add_weight(
            shape = (self.units, self.units),
            initializer = self.kernel_initializer,
            name = 'VAE_mu_kernel',
        )
        
        self.sigma_kernel = self.add_weight(
            shape = (self.units, self.units),
            initializer = self.kernel_initializer,
            name = 'VAE_sigma_kernel',
        )

        self.trans_kernel = self.add_weight(
            shape = (self.units, self.units),
            initializer = self.kernel_initializer,
            name = 'VAE_trans_kernel',            
        )

        self.fc_bias = self.add_weight(
            shape = (self.units),
            initializer = self.bias_initializer,
            name = 'VAE_fc_bias',
        )

        self.mu_bias = self.add_weight(
            shape = (self.units),
            initializer = self.bias_initializer,
            name = 'VAE_mu_bias',
        )

        self.sigma_bias = self.add_weight(
            shape = (self.units),
            initializer = self.bias_initializer,
            name = 'VAE_sigma_bias',
        )

        self.trans_bias = self.add_weight(
            shape = (self.units),
            initializer = self.bias_initializer,
            name = 'VAE_trans_bias',
        )

    def call(self, inputs):
        h = K.bias_add(K.dot(inputs, self.fc_kernel), self.fc_bias)
        relu_h = K.tanh(h)

        self.mu = K.bias_add(K.dot(relu_h, self.mu_kernel), self.mu_bias)
        self.logvar = K.bias_add(K.dot(relu_h, self.sigma_kernel), self.sigma_bias)

        h_z = self.sample_z(self.mu, self.logvar)

        z = K.bias_add(K.dot(h_z, self.trans_kernel), self.trans_bias)
        z = K.tanh(z)

        return z

    def sample_z(self, mu, logvar):
        eps = tf.random.normal(shape = tf.shape(mu))
        return mu + tf.exp(logvar / 2) * eps
    
    def get_loss(self):
        return 0.5 * tf.reduce_sum(tf.exp(self.logvar) + self.mu**2 - 1. - self.logvar, 1)

class AdversarialModel(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 hidden_units,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 name = 'Adversarial_Net',
                 **kwargs):
        super(AdversarialModel, self).__init__(name=name, **kwargs)

        self.units = units
        self.hidden_units = hidden_units

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
    
    def build(self, input_shape):
        input_dim = input_shape[-1][-1]

        self.G_w1 = self.add_weight(
            shape = (input_dim, self.hidden_units),
            initializer = self.kernel_initializer,
            name = 'G_w1'
        )

        self.G_w2 = self.add_weight(
            shape = (self.hidden_units, self.units),
            initializer = self.kernel_initializer,
            name = 'G_w2'
        )

        self.D_w1 = self.add_weight(
            shape = (self.units, self.hidden_units),
            initializer = self.kernel_initializer,
            name = 'D_w1'
        )

        self.D_w2 = self.add_weight(
            shape = (self.hidden_units, 1),
            initializer = self.kernel_initializer,
            name = 'D_w2'
        )

        self.G_b1 = self.add_weight(
            shape = (self.hidden_units),
            initializer = self.bias_initializer,
            name = 'G_b1'
        )

        self.G_b2 = self.add_weight(
            shape = (self.units),
            initializer = self.bias_initializer,
            name = 'G_b2'
        )

        self.D_b1 = self.add_weight(
            shape = (self.hidden_units),
            initializer = self.bias_initializer,
            name = 'D_b1'
        )

        self.D_b2 = self.add_weight(
            shape = (1),
            initializer = self.bias_initializer,
            name = 'D_b2'
        )

    def call(self, inputs):
        (src_enc, tgt_enc) = inputs
        G_sample = self.generator(src_enc)

        real = self.discriminator(tgt_enc)
        fake = self.discriminator(G_sample)

        return real, fake

    def generator(self, src_enc):
        G_h = K.bias_add(K.dot(src_enc, self.G_w1), self.G_b1)
        G_h_relu = tf.nn.relu(G_h)
        G_log_prob = K.bias_add(K.dot(G_h_relu, self.G_w2), self.G_b2)
        G_prob = tf.nn.sigmoid(G_log_prob)

        return G_prob

    def discriminator(self, inputs):
        D_h = K.bias_add(K.dot(inputs, self.D_w1), self.D_b1)
        D_h_relu = tf.nn.relu(D_h)
        D_out = K.bias_add(K.dot(D_h_relu, self.D_w2), self.D_b2)

        return D_out
    
    def clip_D(self):
        self.D_w1.assign(tf.clip_by_value(self.D_w1, -0.01, 0.01))
        self.D_w2.assign(tf.clip_by_value(self.D_w2, -0.01, 0.01))
        self.D_b1.assign(tf.clip_by_value(self.D_b1, -0.01, 0.01))
        self.D_b2.assign(tf.clip_by_value(self.D_b2, -0.01, 0.01))

    def get_generator_loss(self, fake):
        return -tf.reduce_mean(fake)
    
    def get_discriminator_loss(self, real, fake):
        return tf.reduce_mean(fake) - tf.reduce_mean(real)

class AVNMT(tf.keras.layers.Layer):
    def __init__(self,
                 vocabulary,
                 enc_units,
                 dec_units,
                 vae_units,
                 batch_size,
                 embedded_size,
                 output_size,
                 drop_out = 1,
                 max_seq_len = 50,
                 lambda_logits = 1,
                 lambda_vae = 0.2,
                 SOS_ID = 1,
                 EOS_ID = 2,
                 PAD_ID = 0,
                 MASK_ID = 3,
                 UNK_ID = 4,
                 name = 'AVNMT',
                 **kwargs):
        super(AVNMT, self).__init__(name = name, **kwargs)
        self.vocabulary = vocabulary
        self.enc_units = enc_units
        self.dec_units = dec_units
        self.vae_units = vae_units

        self.batch_size = batch_size
        self.embedded_size = embedded_size
        self.output_size = output_size

        self.drop_out = max(0, min(drop_out, 1))
        self.max_seq_len = max_seq_len
        self.lambda_logits = lambda_logits
        self.lambda_vae = lambda_vae

        self.EOS_ID = EOS_ID
        self.PAD_ID = PAD_ID
        self.SOS_ID = SOS_ID
        self.UNK_ID = UNK_ID
        self.MASK_ID = MASK_ID

    def build(self, input_shape):
        self.embedding_softmax_layer = hyper_layer.EmbeddingSharedWeights(
            self.vocabulary, self.embedded_size, pad_id=self.PAD_ID)

        self.src_encoder = core_seq2seq_model.Encoder(self.enc_units, 
                        self.batch_size,
                        self.dec_units, 
                        drop_out = self.drop_out,
                        name = 'src_enc',
                    )

        self.tgt_encoder = core_seq2seq_model.Encoder(self.enc_units, 
                        self.batch_size,
                        self.dec_units, 
                        drop_out = self.drop_out,
                        name = 'tgt_enc',
                    )

        self.decoder = core_seq2seq_model.Decoder(self.dec_units, 
                        self.embedded_size, 
                        self.output_size,
                        drop_out = self.drop_out,
                    )

        self.VAE = VAE(self.vae_units)

        self.AN = AdversarialModel(units = 2 * self.enc_units, hidden_units = self.enc_units)
    
    def call(self, 
             inputs, 
             teacher_force_rate = 0.5, 
             sample_force_rate = 0.8,
             inference = False,
             use_while_loop = True):
        (src, tgt) = inputs
        mask = tf.math.logical_not(tf.math.equal(tgt, self.PAD_ID))
        tgt = tf.pad(tgt, [[0, 0], [1, 0]],
                        constant_values=self.SOS_ID)
        
        src = self.embedding_softmax_layer(src)
        tgt = self.embedding_softmax_layer(tgt)

        tgt_seq_len = tgt.shape[1]

        src_enc, hidden = self.src_encoder(src)
        tgt_enc, tgt_hidden = self.tgt_encoder(tgt[:, 1:, :])

        src_h = tf.reduce_mean(src_enc, axis = 1)
        tgt_h = tf.reduce_mean(tgt_enc, axis = 1)

        real, fake = self.AN((src_h, tgt_h))
        
        sample_force = random.random() < sample_force_rate 
        if sample_force:
            vae_input = tf.concat([src_h, tgt_h], axis = -1)
        else:
            sample_h = self.AN.generator(src_h)
            vae_input = tf.concat([src_h, sample_h], axis = -1)
            
        h_z = self.VAE(vae_input)
        h_z = tf.expand_dims(h_z, axis = 1)

        outputs = []
        output = tgt[:, 0, :]
        output = tf.expand_dims(output, axis = 1)

        states = tf.zeros_like(hidden)

        if use_while_loop:
            i = 1
            cond = lambda i, tgt, inputs, h_z, src_enc, hidden, states, outputs, teacher_force_rate: i < tgt_seq_len
            tf.while_loop(cond, self.loop_body, loop_vars = [i, tgt, output, h_z, src_enc, hidden, states, outputs, teacher_force_rate])

        else:
            for i in range(1, tgt_seq_len):
                output = tf.concat([output, h_z], axis = -1)
                dec_output, hidden, states = self.decoder(output, 
                                    initial_state = [hidden, states], 
                                    context = src_enc,
                                )
                outputs.append(dec_output)

                teacher_force = random.random() < teacher_force_rate     
                if teacher_force:
                    output = tgt[:, i, :]
                    output = tf.expand_dims(output, axis = 1)
                else:
                    output = self.embedding_softmax_layer(tf.argmax(dec_output, axis=-1))

        logits = tf.concat(outputs, axis = 1)
        # logits = tf.transpose(logits, [1, 0, 2])

        mask = tf.cast(mask, dtype=logits.dtype)
        mask = tf.tile(tf.expand_dims(mask, axis = -1), [1, 1, self.output_size])
        logits *= mask

        return logits, real, fake

    def inference(self, inputs):
        src = self.embedding_softmax_layer(inputs)

        enc_output, hidden = self.src_encoder(src)
        
        src_h = tf.reduce_mean(enc_output, axis = 1)

        example = self.AN.generator(src_h)


        vae_input = tf.concat([src_h, example], axis = -1)
        h_z = self.VAE(vae_input)
        h_z = tf.expand_dims(h_z, axis = 1)

        outputs = self.predict(initial_state=hidden, h_z = h_z, context = enc_output)

        return outputs

    def get_predict_with_logits(self, logits):
        return tf.argmax(logits, axis=-1, output_type=tf.int32)

    def loop_body(self, i, tgt, inputs, h_z, src_enc, hidden, states, outputs, teacher_force_rate):
        inputs = tf.concat([inputs, h_z], axis = -1)
        dec_output, hidden, states = self.decoder(inputs, 
                        initial_state = [hidden, states], 
                        context = src_enc,
                    )
        outputs.append(dec_output)

        teacher_force = random.random() < teacher_force_rate     
        if teacher_force:
            inputs = tgt[:, i, :]
            inputs = tf.expand_dims(inputs, axis = 1)
        else:
            inputs = self.embedding_softmax_layer(tf.argmax(dec_output, axis=-1)) 

        return i + 1, tgt, inputs, h_z, src_enc, hidden, states, outputs, teacher_force_rate

    def predict(self,
                initial_state=None,
                h_z=None,
                context=None,
                training=False):
        def _loop_fn(inputs, i, cache):
            inputs = tf.expand_dims(inputs[:, i], -1)
            inputs = self.embedding_softmax_layer(inputs)

            inputs = tf.concat([inputs, h_z], axis = -1)

            logits, hidden, states = self.decoder(
                inputs,
                context=cache['context'],
                initial_state=cache['initial_state'],
                training=training)

            cache['initial_state'] = [hidden, states]
            # logits = self.embedding_softmax_layer(logits, linear=True)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache
        
        batch_size = tf.shape(context)[0]
        max_decode_length = self.max_seq_len
        initial_ids = tf.zeros([batch_size], dtype=tf.int32) + self.SOS_ID
        h_z = tf.tile(h_z, [4, 1, 1])
        initial_c = tf.keras.backend.zeros_like(initial_state)
        initial_state = (initial_state, initial_c)

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
    
    def get_logits_loss(self, labels, logits):
        logits = tf.reshape(logits, [-1, self.output_size])
        labels = tf.reshape(labels, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        mask = tf.math.logical_not(tf.math.equal(labels, 0))
        mask = tf.cast(mask, dtype=loss.dtype)

        return tf.reduce_mean(loss * mask)    
    
    def clip_discriminator(self):
        self.AN.clip_D()

    def set_lambda_vae(self, value):
        self.lambda_vae = value

    def set_lambda_logits(self, value):
        self.lambda_logits = value

    def get_vae_loss(self):
        return tf.reduce_mean(self.VAE.get_loss())

    def get_generator_loss(self, fake):
        return self.AN.get_generator_loss(fake)

    def get_discriminator_loss(self, real, fake):
        return self.AN.get_discriminator_loss(real, fake)
    
    def get_model_loss(self, labels, logits, real, fake):
        logits_loss = self.get_logits_loss(labels, logits)
        vae_loss = self.get_vae_loss()
        D_loss = self.get_discriminator_loss(real, fake)
        G_loss = self.get_generator_loss(fake)

        return self.lambda_logits * logits_loss + self.lambda_vae * vae_loss, logits_loss, vae_loss, D_loss, G_loss

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    src = tf.constant(range(10), shape = (2, 5), dtype=tf.int32)
    tgt = tf.constant(range(10), shape = (2, 5), dtype=tf.int32)

    model = AVNMT(100, 16, 16, 32, 5, 100, 100)
    logits, real, fake = model((src, tgt))

    print('vae loss is: {}'.format(model.get_vae_loss()))
    print('logits loss is: {}'.format(model.get_logits_loss(tgt, logits)))
    print('D loss is: {}'.format(model.get_discriminator_loss(fake)))
    print('G loss is: {}'.format(model.get_generator_loss(real, fake)))
    # z = model.inference((src))
    # print(z)
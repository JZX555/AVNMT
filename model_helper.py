import tensorflow as tf

def dropout_mask_helper(ones, rate, training=None, count=1):
    def dropped_inputs():
        return tf.keras.backend.dropout(ones, rate)

    if count > 1:
        return [
            tf.keras.backend.in_train_phase(
                dropped_inputs, ones, training=training) for _ in range(count)
        ]
    return tf.keras.backend.in_train_phase(
        dropped_inputs, ones, training=training)
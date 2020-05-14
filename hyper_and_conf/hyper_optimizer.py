import numpy as np
import tensorflow as tf
import re
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
K = tf.keras.backend


class LazyAdam(tf.keras.optimizers.Adam):
    """Variant of the Adam optimizer that handles sparse updates more efficiently.

  The original Adam algorithm maintains two moving-average accumulators for
  each trainable variable; the accumulators are updated at every step.
  This class provides lazier handling of gradient updates for sparse
  variables.  It only updates moving-average accumulators for sparse variable
  indices that appear in the current batch, rather than updating the
  accumulators for all indices. Compared with the original Adam optimizer,
  it can provide large improvements in model training throughput for some
  applications. However, it provides slightly different semantics than the
  original Adam algorithm, and may lead to different empirical results.
  Note, amsgrad is currently not supported and the argument can only be
  False.

  This class is borrowed from:
  https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lazy_adam.py
  """
    def _resource_apply_sparse(self, grad, var, indices):
        """Applies grad for one step."""
        # file_writer = tf.summary.create_file_writer('/Users/barid/Documents/workspace/alpha/lip_read/model_summary' + "/gradient")
        # file_writer.set_as_default()
        tf.summary.histogram('gradient', grad)
        # tf.summary.merge_all()
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.math.pow(beta_1_t, local_step)
        beta_2_power = tf.math.pow(beta_2_t, local_step)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        lr = (lr_t * tf.math.sqrt(1 - beta_2_power) / (1 - beta_1_power))

        # \\(m := beta1 * m + (1 - beta1) * g_t\\)
        m = self.get_slot(var, 'm')
        m_t_slice = beta_1_t * tf.gather(m, indices) + (1 - beta_1_t) * grad

        m_update_kwargs = {
            'resource': m.handle,
            'indices': indices,
            'updates': m_t_slice
        }
        m_update_op = tf.raw_ops.ResourceScatterUpdate(**m_update_kwargs)

        # \\(v := beta2 * v + (1 - beta2) * (g_t * g_t)\\)
        v = self.get_slot(var, 'v')
        v_t_slice = (beta_2_t * tf.gather(v, indices) +
                     (1 - beta_2_t) * tf.math.square(grad))

        v_update_kwargs = {
            'resource': v.handle,
            'indices': indices,
            'updates': v_t_slice
        }
        v_update_op = tf.raw_ops.ResourceScatterUpdate(**v_update_kwargs)

        # \\(variable -= learning_rate * m_t / (epsilon_t + sqrt(v_t))\\)
        var_slice = lr * m_t_slice / (tf.math.sqrt(v_t_slice) + epsilon_t)

        var_update_kwargs = {
            'resource': var.handle,
            'indices': indices,
            'updates': var_slice
        }
        var_update_op = tf.raw_ops.ResourceScatterSub(**var_update_kwargs)

        return tf.group(*[var_update_op, m_update_op, v_update_op])


class LearningRateFn(object):
    """Creates learning rate function."""
    def __init__(self, learning_rate, hidden_size, warmup_steps):
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.warmup_steps = float(warmup_steps)

    def __call__(self, global_step):
        """Calculate learning rate with linear warmup and rsqrt decay."""
        step = float(global_step)
        learning_rate = self.learning_rate
        learning_rate *= (self.hidden_size**-0.5)
        # Apply linear warmup
        learning_rate *= np.minimum(1, step / self.warmup_steps)
        # # Apply rsqrt decay
        learning_rate /= np.sqrt(np.maximum(step, self.warmup_steps))
        # ratio = 30
        # cut_frac = 0.3
        # cut = int(3000 * cut_frac)
        #
        # if step < cut:
        #     p = step / cut
        # else:
        #     p = 1 - ((step - cut) / (cut * (ratio - 1)))
        # learning_rate = 0.001 * (1 + p * (ratio - 1)) / ratio
        return learning_rate


class LearningRateScheduler(tf.keras.callbacks.Callback):
    """Keras callback to schedule learning rate.

  TODO(tianlin): Refactor this scheduler and LearningRateBatchScheduler in
  official/resnet/keras/keras_common.py.
  """
    def __init__(self, schedule, init_steps=0, verbose=False):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        if init_steps is None:
            init_steps = 0.0
        self.steps = float(init_steps)  # Total steps during training.

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if not hasattr(self.model.optimizer, 'iterations'):
            raise ValueError('Optimizer must have a "iterations" attribute.')

    def on_train_batch_begin(self, batch, logs=None):
        """Adjusts learning rate for each train batch."""
        if self.verbose > 0:
            iterations = K.get_value(self.model.optimizer.iterations)
            print('Original iteration %d' % iterations)

        self.steps += 1.0
        try:  # new API
            lr = float(K.get_value(self.model.optimizer.lr))
            lr = self.schedule(self.steps, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(self.steps)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        K.set_value(self.model.optimizer.iterations, self.steps)

        if self.verbose > 0:
            print(
                'Batch %05d Step %05d: LearningRateScheduler setting learning '
                'rate to %s.' % (batch + 1, self.steps, lr))

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        # logs['steps'] = self.steps

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        logs['steps'] = self.steps


class LearningRateVisualization(tf.keras.callbacks.Callback):
    """Keras callback to schedule learning rate.

  TODO(tianlin): Refactor this scheduler and LearningRateBatchScheduler in
  official/resnet/keras/keras_common.py.
  """
    def __init__(self, init_steps=0, verbose=False):
        super(LearningRateVisualization, self).__init__()
        self.verbose = verbose
        self.steps = float(init_steps)  # Total steps during training.

    # def on_epoch_begin(self, epoch, logs=None):
    # if not hasattr(self.model.optimizer, 'lr'):
    #     raise ValueError('Optimizer must have a "lr" attribute.')
    # if not hasattr(self.model.optimizer, 'iterations'):
    #     raise ValueError('Optimizer must have a "iterations" attribute.')

    def on_train_batch_begin(self, batch, logs=None):
        """Adjusts learning rate for each train batch."""
        # if self.verbose > 0:
        #     iterations = K.get_value(self.model.optimizer.iterations)
        #     print('Original iteration %d' % iterations)

        self.steps += 1.0
        # lr = float(K.get_value(self.model.optimizer.lr))
        # try:  # new API
        #     lr = float(K.get_value(self.model.optimizer.lr))
        #     lr = self.schedule(self.steps, lr)
        # except TypeError:  # Support for old API for backward compatibility
        #     lr = self.schedule(self.steps)
        # if not isinstance(lr, (float, np.float32, np.float64)):
        #     raise ValueError('The output of the "schedule" function '
        #                      'should be float.')
        # K.set_value(self.model.optimizer.lr, lr)
        # K.set_value(self.model.optimizer.iterations, self.steps)

        # if self.verbose > 0:
        #     print(
        #         'Batch %05d Step %05d: LearningRateScheduler setting learning '
        #         'rate to %s.' % (batch + 1, self.steps, lr))

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr.current_lr)
        # logs['steps'] = self.steps

    # def on_epoch_end(self, epoch, logs=None):
    #     logs = logs or {}
    #     logs['lr'] = K.get_value(self.model.optimizer.lr)
    #     logs['steps'] = self.steps


class warmup_lr(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applys a warmup schedule on a given learning rate decay schedule."""
    def __init__(self,
                 initial_learning_rate,
                 warmup_steps,
                 decay_steps,
                 end_learning_rate=0.000001,
                 cycle=False,
                 power=4.0,
                 name=None):
        # super(WarmUp, self).__init__()
        self.initial_learning_rate = self.current_lr = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.cycle = cycle
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or 'WarmUp'):
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = step * 1.0
            warmup_steps_float = self.warmup_steps * 1.0
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * warmup_percent_done**self.power
            return tf.cond(global_step_float < warmup_steps_float,
                           lambda: warmup_learning_rate,
                           lambda: self.polynomialDecay(step),
                           name='dynamic_lr')
            # if global_step_float < warmup_steps_float:
            #     return warmup_learning_rate
            # else:
            #     return self.polynomialDecay(step)

    def polynomialDecay(self, step):

        global_step_recomp = tf.minimum(step, self.decay_steps)

        p = global_step_recomp / self.decay_steps
        lr = (self.initial_learning_rate - self.end_learning_rate) * (
            1 - p)**self.power + self.end_learning_rate
        return lr

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_schedule_fn': self.decay_schedule_fn,
            'warmup_steps': self.warmup_steps,
            'power': self.power,
            "decay_steps": self.decay_steps,
            "end_learning_rate": self.end_learning_rate,
            "cycle": self.cycle,
            "name": self.name
        }

class warmup_lr_obj():
    """Applys a warmup schedule on a given learning rate decay schedule."""
    def __init__(self,
                 initial_learning_rate,
                 warmup_steps,
                 decay_steps,
                 end_learning_rate=0.000001,
                 cycle=False,
                 power=4.0,
                 name=None):
        # super(WarmUp, self).__init__()
        self.initial_learning_rate = self.current_lr = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.cycle = cycle
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or 'WarmUp'):
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = step * 1.0
            warmup_steps_float = self.warmup_steps * 1.0
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * warmup_percent_done**self.power
            if global_step_float < warmup_steps_float:
                return warmup_learning_rate
            else:
                return self.polynomialDecay(step)
            # if global_step_float < warmup_steps_float:
            #     return warmup_learning_rate
            # else:
            #     return self.polynomialDecay(step)

    def polynomialDecay(self, step):

        global_step_recomp = min(step, self.decay_steps)

        p = global_step_recomp / self.decay_steps
        lr = (self.initial_learning_rate - self.end_learning_rate) * (
            1 - p)**self.power + self.end_learning_rate
        return lr

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_schedule_fn': self.decay_schedule_fn,
            'warmup_steps': self.warmup_steps,
            'power': self.power,
            "decay_steps": self.decay_steps,
            "end_learning_rate": self.end_learning_rate,
            "cycle": self.cycle,
            "name": self.name
        }

def create_lr_fn(init_lr=0.001, num_train_steps=10000, num_warmup_steps=50):
    """Creates an optimizer with learning rate schedule."""
    # Implements linear decay of the learning rate.
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr,
        decay_steps=num_train_steps,
        end_learning_rate=0.000001)
    learning_rate_fn = warmup_lr(initial_learning_rate=init_lr,
                                 decay_schedule_fn=learning_rate_fn,
                                 warmup_steps=num_warmup_steps)
    return learning_rate_fn


def create_optimizer(init_lr, num_train_steps, num_warmup_steps):
    """Creates an optimizer with learning rate schedule."""
    # Implements linear decay of the learning rate.
    # learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    #     initial_learning_rate=init_lr,
    #     decay_steps=num_train_steps,
    #     end_learning_rate=0.0)
    # if num_warmup_steps:
    learning_rate_fn = warmup_lr(initial_learning_rate=init_lr,
                                 warmup_steps=num_warmup_steps,
                                 decay_steps=num_train_steps)
    return AdamWeightDecay(learning_rate=learning_rate_fn,
                           weight_decay_rate=0.01,
                           beta_1=0.9,
                           beta_2=0.999,
                           epsilon=1e-6,
                           exclude_from_weight_decay=['layer_norm', 'bias'])


class AdamWeightDecay(tf.keras.optimizers.Adam):
    """Adam enables L2 weight decay and clip_by_global_norm on gradients.

  Just adding the square of the weights to the loss function is *not* the
  correct way of using L2 regularization/weight decay with Adam, since that will
  interact with the m and v parameters in strange ways.

  Instead we want ot decay the weights in a manner that doesn't interact with
  the m/v parameters. This is equivalent to adding the square of the weights to
  the loss with plain (non-momentum) SGD.
  """
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 weight_decay_rate=0.0,
                 include_in_weight_decay=None,
                 exclude_from_weight_decay=None,
                 name='AdamWeightDecay',
                 **kwargs):
        super(AdamWeightDecay, self).__init__(learning_rate, beta_1, beta_2,
                                              epsilon, amsgrad, name, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay

    @classmethod
    def from_config(cls, config):
        """Creates an optimizer from its config with WarmUp custom object."""
        custom_objects = {'WarmUp': warmup_lr}
        return super(AdamWeightDecay,
                     cls).from_config(config, custom_objects=custom_objects)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype,
                                                    apply_state)
        apply_state['weight_decay_rate'] = tf.constant(
            self.weight_decay_rate, name='adam_weight_decay_rate')

    def _decay_weights_op(self, var, learning_rate, apply_state):
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            return var.assign_sub(learning_rate * var *
                                  apply_state['weight_decay_rate'],
                                  use_locking=self._use_locking)
        return tf.no_op()

    def apply_gradients(self, grads_and_vars, name=None):
        grads, tvars = list(zip(*grads_and_vars))
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        return super(AdamWeightDecay, self).apply_gradients(zip(grads, tvars))

    def _get_lr(self, var_device, var_dtype, apply_state):
        """Retrieves the learning rate with the given state."""
        if apply_state is None:
            return self._decayed_lr_t[var_dtype], {}

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients

        return coefficients['lr_t'], dict(apply_state=apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype,
                                    apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay,
                         self)._resource_apply_dense(grad, var, **kwargs)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype,
                                    apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay,
                         self)._resource_apply_sparse(grad, var, indices,
                                                      **kwargs)

    def get_config(self):
        config = super(AdamWeightDecay, self).get_config()
        config.update({
            'weight_decay_rate': self.weight_decay_rate,
        })
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.weight_decay_rate == 0:
            return False

        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True

        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

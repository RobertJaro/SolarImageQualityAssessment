import logging
from multiprocessing import Queue, Process

import numpy as np
import tensorflow as tf
from keras import initializers, regularizers, constraints, backend as K
from keras.activations import softmax
from keras.engine import Layer, InputSpec
from keras.layers import ZeroPadding2D
from keras.layers.merge import _Merge
from sklearn.utils import shuffle


def write(data_generator, tasks, processed):
    while True:
        idx = tasks.get()
        try:
            processed.put(data_generator[idx])
        except Exception as ex:
            logging.error("Exception during loading data (%d): %s" % (idx, ex))


class Enqueuer():
    def __init__(self, data_generator, batch_size=1, n_workers=16, max_queue=128, shuffle=True):
        self.tasks = Queue()
        self.processed = Queue(max_queue)
        self.data_generator = data_generator
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.processes = []
        self.shuffle = shuffle

        self.loadEpoch()

        for _ in range(n_workers):
            p = Process(target=write, args=(data_generator, self.tasks, self.processed))
            self.processes.append(p)
            p.start()

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            item = self.processed.get()
            if not isinstance(item, tuple):
                item = (item,)
            batch.append(item)
            if self.tasks.qsize() == 0 and self.processed.qsize() == 0:  # start of next epoch
                self.loadEpoch()
                break  # finish batch and start next epoch
        result = [np.array(list(i)) for i in zip(*batch)]
        return result[0] if len(result) == 1 else result

    def loadEpoch(self):
        sequence = list(range(len(self.data_generator)))
        sequence = shuffle(sequence) if self.shuffle else sequence
        list(map(self.tasks.put, sequence))

    def __len__(self):
        return int(np.ceil(len(self.data_generator) / self.batch_size))

    def stop(self):
        for p in self.processes:
            p.terminate()


class NameLayer(Layer):

    def call(self, inputs, **kwargs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _merge_function(self, inputs):
        weights = K.random_uniform((K.shape(inputs[0])[0], 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


class QuantizationLayer(Layer):
    def __init__(self, centers=5, **kwargs):
        self.centers = centers
        super().__init__(**kwargs)

    def call(self, input, **kwargs):
        # input sigmoid scaled [0,1]
        centers = tf.cast(tf.range(self.centers), tf.float32)

        w_stack = tf.stack([input * (self.centers - 1) for _ in range(self.centers)], axis=-1)
        w_hard = tf.cast(tf.argmin(tf.abs(w_stack - centers), axis=-1), tf.float32)

        smx = softmax(-1.0 * tf.abs(w_stack - centers), axis=-1)
        w_soft = tf.einsum('ijklm,m->ijkl', smx, centers)

        w_bar = tf.stop_gradient(w_hard - w_soft) + w_soft

        return w_bar


class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """

    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ReflectionPadding2D(ZeroPadding2D):
    def call(self, x, mask=None):
        pattern = [[0, 0],
                   self.padding[0],
                   self.padding[1],
                   [0, 0]]
        return tf.pad(x, pattern, mode='REFLECT')

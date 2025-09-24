import tensorflow as tf
import keras

BATCH_AXIS = 0
FEATURES_AXIS = 1
TIME_AXIS = 2
WIDTH_AXIS = 3
HEIGHT_AXIS = 4

FILTERS = [16, 32, 64, 128]
SIZES = [26, 12, 5, 2]

class D3DBasicBlock(keras.layers.Layer):
    def __init__(self, name, filters=0):
        """args:
            - filters should be an int corresponding to the index in FILTERS
            - block=1 means this is the second block with the same number of filters"""
        super().__init__(name=name)

        self.conv1 = keras.layers.Conv3D(filters=FILTERS[filters], kernel_size=(2,3,3), strides=(1,1,1), data_format="channels_first")
        self.conv2 = keras.layers.Conv3D(filters=FILTERS[filters], kernel_size=(1,3,3), strides=(1,1,1), data_format="channels_first")

        self.pad = keras.layers.ZeroPadding3D(padding=(0,1,1), data_format="channels_first")
        self.bn = keras.layers.BatchNormalization(axis=FEATURES_AXIS)
        self.relu = keras.layers.ReLU()

        self.cat = keras.layers.Concatenate(axis=TIME_AXIS)

    def build(self, input_shape):
        self.cache = self.add_weight(
            name="cache",
            shape=input_shape[1:],
            initializer="zeros",
            trainable=False,
        )

    def call(self, x):
        # Cache x for the residual connections
        tmp_cache = x
        tmp_cache2 = tf.expand_dims(self.cache, axis=0)
        x = self.cat([tmp_cache2, x])
        self.cache.assign(tf.reduce_mean(tmp_cache, axis=0))

        # First convolution, BN and ReLU
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        # Second convolution, BN
        x = self.pad(x)
        x = self.conv2(x)
        x = self.bn(x)

        # Residual connection and ReLU
        x = x + tmp_cache
        x = self.relu(x)

        return x


class D3DNetwork(keras.Model):
    def __init__(self, base: keras.Model, skip_steps: int, *args, **kwargs):
        super().__init__(*args, inputs=base.inputs, outputs=base.outputs, **kwargs)
        self.skip_steps = skip_steps
        self._step = self.add_weight(
            shape=(),
            initializer='zeros',
            dtype=tf.int64,
            trainable=False,
            name="_step",
        )

    def train_step(self, data):
        self._step.assign_add(1)

        def _do_update():
            # If we've taken enough steps, just fall back on the standard train_step
            return super(D3DNetwork, self).train_step(data)

        def _do_skip():
            # Compute the prediction and metrics, but don't backprop
            x, y, _ = keras.utils.unpack_x_y_sample_weight(data)
            y_pred = self(x, training=False)    # or skip this for zero comp
            loss_dtype = self.dtype_policy.compute_dtype if hasattr(self, "dtype_policy") else tf.float32
            logs = {"loss": tf.zeros((), dtype=loss_dtype)}
            for metric in self.metrics:
                metric.update_state(y, y_pred)

                # If the metric isn't a nested list of metrics, just add it
                if type(metric.result()) != dict:
                    logs.update({metric.name: metric.result()})
                # Otherwise, add all of the nested metrics
                else:
                    logs.update(**metric.result())
            return logs

        # For the first skip_steps number of frames, skip gradient computation and backprop
        # so we can effectively populate the caches before training starts
        # This checks the condition and does _do_update if it is True, otherwise it skips
        return tf.cond(tf.greater(self._step, self.skip_steps), _do_update, _do_skip)

def construct_model(skip_steps=20):
    input_frame = keras.Input(shape=(1,1,224,224), name="frame")

    conv1 = keras.layers.Conv3D(filters=16, kernel_size=(1,7,7), strides=(1,2,2), data_format="channels_first")(input_frame)
    pool = keras.layers.MaxPool3D(pool_size=(1,3,3), strides=(1,2,2), data_format="channels_first")(conv1)

    conv2_0 = D3DBasicBlock("2_0", filters=0)(pool)
    conv2_1 = D3DBasicBlock("2_1", filters=0)(conv2_0)
    conv2_2 = keras.layers.Conv3D(filters=32, kernel_size=(1,3,3), strides=(1,2,2), data_format="channels_first")(conv2_1)

    conv3_0 = D3DBasicBlock("3_0", filters=1)(conv2_2)
    conv3_1 = D3DBasicBlock("3_1", filters=1)(conv3_0)
    conv3_2 = keras.layers.Conv3D(filters=64, kernel_size=(1,3,3), strides=(1,2,2), data_format="channels_first")(conv3_1)
    
    conv4_0 = D3DBasicBlock("4_0", filters=2)(conv3_2)
    conv4_1 = D3DBasicBlock("4_1", filters=2)(conv4_0)
    conv4_2 = keras.layers.Conv3D(filters=128, kernel_size=(1,3,3), strides=(1,2,2), data_format="channels_first")(conv4_1)

    conv5_0 = D3DBasicBlock("5_0", filters=3)(conv4_2)
    conv5_1 = D3DBasicBlock("5_1", filters=3)(conv5_0)

    flat = keras.layers.Flatten(data_format="channels_first")(conv5_1)
    output = keras.layers.Dense(1)(flat)

    base_model = keras.Model(input_frame, output, name="D3D")
    d3d_model = D3DNetwork(base_model, skip_steps)

    return d3d_model

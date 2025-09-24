import random
from collections import defaultdict
import xml.etree.ElementTree as ET
import imageio.v3 as iio
import tensorflow as tf
import keras
import numpy as np
import datetime
from sklearn.model_selection import train_test_split

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

def construct_model():
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

    model = keras.Model(input_frame, output, name="D3D")

    return model


def dataset_gen(frames, y):
    # 10 sub-videos (randomly ordered)
    frames_ = np.array_split(frames, 10, axis=0,)
    y_ = np.array_split(y, 10)
    data_ = list(zip(frames_, y_))
    random.shuffle(data_)
    for (data,label) in data_:
        yield(data,label)



if __name__ == "__main__":
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = construct_model()
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy', keras.metrics.AUC()],
    )
    # model.load_weights(filepath="model.keras")
    model.load_weights(filepath="model_checkpoints/model2025-09-23_11-00-35.keras")

    # Load the labels
    tree = ET.parse("finger_tap_training_data/annotations.xml")
    root = tree.getroot()
    labels = defaultdict(list)

    for image in root.findall(".//image"):
        task_id = image.attrib["task_id"]
        # frame_id = int(image.attrib["id"])
        has_tag = image.find("tag") is not None
        # frame_ids.append(frame_id)
        labels[task_id].append(1 if has_tag else 0)

    for v in model.non_trainable_variables:
        if "cache" in v.name:
            v.assign(np.zeros_like(v))

    test_video = "finger_tap_training_data/picamera0_2025-09-19_12-05-47.mp4"
    test_data = iio.imread(test_video, index=None)
    frame_crop = np.random.randint(low=0, high=100, size=(2,))
    test_data = np.dot(test_data, [0.2989, 0.5870, 0.1140]) # grayscale
    test_data = test_data.reshape(-1,1,1,324,324)[...,frame_crop[0]:frame_crop[0]+224,frame_crop[1]:frame_crop[1]+224]
    test_y = labels['6']

    # ROC
    # preds = model.predict(test_data, batch_size=1)
    # auc = keras.metrics.auc(from_logits=true, num_thresholds=200)
    # auc.update_state(test_y, preds)
    # roc = auc.result()
    # print(roc)

    results = model.evaluate(
            test_data,
            np.array(test_y),
            batch_size=1,
            callbacks=[tensorboard_callback],
    )
    print(results)

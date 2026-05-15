import random
from collections import defaultdict
import xml.etree.ElementTree as ET
import imageio.v3 as iio
import tensorflow as tf
import keras
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from d3d_network import construct_model


if __name__ == "__main__":
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = construct_model(skip_steps=0)
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy', keras.metrics.AUC()],
    )
    # model.load_weights(filepath="model.keras")
    model.load_weights(filepath="model_checkpoints/model2025-09-25_09-06-49.keras")

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

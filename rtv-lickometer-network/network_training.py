from collections import defaultdict
from copy import copy
import datetime
import os
import random

from d3d_network import construct_model

import xml.etree.ElementTree as ET
import imageio.v3 as iio
import keras
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from vidaug import augmentors as va

def dataset_gen(data, y):
    data_orig = copy(data)
    for aug_num in range(50):
        # Perform the sequence of augmentations and stack to a single grayscale array
        data = aug_seq(data_orig)
        data = np.stack(data, axis=0)
        data = np.dot(data, [0.2989, 0.5870, 0.1140]) # grayscale
        # split to sub-videos (randomly ordered)
        n_splits = 5
        data_ = np.array_split(data, n_splits, axis=0,)
        y_ = np.array_split(y, n_splits)
        data = list(zip(data_, y_))
        random.shuffle(data)
        for (d,l) in data:
            yield(d,l,aug_num)


def main():
    # For performance tracking
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = construct_model(skip_steps=100)
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=True,),
        metrics=['accuracy', keras.metrics.AUC(name="auc")],
    )

    # Load the labels
    tree = ET.parse("finger_tap_training_data/annotations.xml")
    root = tree.getroot()

    # Frame info
    # frame_ids = []
    labels = defaultdict(list)

    for image in root.findall(".//image"):
        task_id = image.attrib["task_id"]
        # frame_id = int(image.attrib["id"])
        has_tag = image.find("tag") is not None
        # frame_ids.append(frame_id)
        labels[task_id].append(1 if has_tag else 0)

    # Load the video
    videos = ["finger_tap_training_data/picamera0_2025-09-19_11-59-22.mp4", "finger_tap_training_data/picamera0_2025-09-19_12-00-59.mp4", "finger_tap_training_data/picamera0_2025-09-19_12-02-28.mp4", "finger_tap_training_data/picamera0_2025-09-19_12-04-21.mp4",]

    # Load weights from a previous run
    # model.load_weights("model.keras")

    # track if we are getting better at the test video, if not, stop training
    # prev_test_loss = np.inf
    # curr_test_loss = 1e6
    # while prev_test_loss > curr_test_loss:

    for idx,video in enumerate(videos):
        print(f"Training video #{idx+1}", flush=True)
        label_idx = idx + 2 # the task labels in cvat started at 2 (because I messed up number 1)
        y = np.array(labels[str(label_idx)])
        data = iio.imread(video, index=None)

        # Split the videos into smaller segments (10 per video)
        # and augment in 10 different ways
        aug_num_old = -1
        for (data_inner,y_inner,aug_num) in dataset_gen(data,y):
            # If we are on to a new augmentation, reset the cache
            if aug_num > aug_num_old:
                print(f"Augmentation #{aug_num+1}", flush=True)
                for v in model.non_trainable_variables:
                    if "cache" in v.name:
                        v.assign(np.zeros_like(v))
                    elif "_step" in v.name:
                        v.assign(0)
                aug_num_old = aug_num
                iio.imwrite(f"augmented_videos/video{idx+1}_aug{aug_num+1}.mp4", data_inner)
            data_inner = data_inner.reshape(-1,1,1,224,224)
            # We don't want to shuffle because the temporal information in adjacent frames is important
            # So just validate on the last 20% of frames
            (train_data, valid_data, train_y, valid_y) = train_test_split(data_inner, y_inner, shuffle=False, test_size=0.2)
            model.fit(
                    train_data.reshape(-1,1,1,224,224),
                    np.array(train_y), 
                    epochs=1,
                    batch_size=1,
                    validation_data=(valid_data.reshape(-1,1,1,224,224), valid_y),
                    callbacks=[tensorboard_callback],
                    class_weight={0: 1.0, 1: 100.0},
            )

        print(f"Training on full video (#{idx+1})", flush=True)
        for v in model.non_trainable_variables:
            if "cache" in v.name:
                v.assign(np.zeros_like(v))
            elif "_step" in v.name:
                v.assign(0)

        # Perform the sequence of augmentations and stack to a single grayscale array
        data = aug_seq(data)
        data = np.stack(data, axis=0)
        data = np.dot(data, [0.2989, 0.5870, 0.1140]) # grayscale

        model.fit(
                data.reshape(-1,1,1,224,224),
                np.array(y), 
                epochs=1,
                batch_size=1,
                # validation_data=(valid_data, valid_y),
                callbacks=[tensorboard_callback],
                class_weight={0: 1.0, 1: 100.0},
        )

    # save the trained model
    if not os.path.isdir("model_checkpoints"):
        os.mkdir("model_checkpoints")
    time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model.save(filepath=f"model_checkpoints/model{time_now}.keras")
    # model.save(filepath="model.keras")
    model.save(filepath=f"model_checkpoints/model{time_now}.h5")

    for v in model.non_trainable_variables:
        if "cache" in v.name:
            v.assign(np.zeros_like(v))
        elif "_step" in v.name:
            v.assign(0)

    test_video = "finger_tap_training_data/picamera0_2025-09-19_12-05-47.mp4"
    test_data = iio.imread(test_video, index=None)
    # frame_crop = np.random.randint(low=0, high=100, size=(2,))
    frame_crop = [50, 50] # user-defined crop from training video gathering
    test_data = np.dot(test_data, [0.2989, 0.5870, 0.1140]) # grayscale
    test_data = test_data.reshape(-1,1,1,324,324)[...,frame_crop[0]:frame_crop[0]+224,frame_crop[1]:frame_crop[1]+224]
    test_y = labels['6']

    # prev_test_loss = curr_test_loss
    curr_test_loss = model.evaluate(
            test_data,
            np.array(test_y),
            batch_size=1,
            callbacks=[tensorboard_callback],
    )
    model.save(filepath="model.keras")


if __name__ == "__main__":
    # Using vidaug package for video augmentation
    sometimes = lambda aug: va.Sometimes(0.5, aug)
    aug_seq = va.Sequential(
        [
            va.RandomCrop(size=(224,224)),
            va.RandomRotate(degrees=45),
            sometimes(va.HorizontalFlip()),
            # sometimes(va.VerticalFlip()),
            # sometimes(va.Pepper()),
        ]
    )
   
    main()

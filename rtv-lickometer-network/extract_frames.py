import xml.etree.ElementTree as ET
import imageio.v3 as iio
import numpy as np

frames = {}
# videos = ["finger_tap_training_data/picamera0_2025-09-19_11-59-22.mp4", "finger_tap_training_data/picamera0_2025-09-19_12-00-59.mp4", "finger_tap_training_data/picamera0_2025-09-19_12-02-28.mp4", "finger_tap_training_data/picamera0_2025-09-19_12-04-21.mp4", "finger_tap_training_data/picamera0_2025-09-19_12-05-47.mp4",]
videos = ["finger_tap_training_data/picamera0_2025-09-19_12-05-47.mp4"]
for idx,video in enumerate(videos):
    idx = idx + 2 # the task labels in cvat started at 2 (because I messed up number 1)
    frames_ = iio.imread(video, index=None)   
    frames[idx] = frames_

for k,v in frames.items():
    fname = videos[k-2]
    np.save(fname.split('.mp4')[0]+'.npy', v)


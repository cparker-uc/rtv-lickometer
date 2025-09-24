from collections import defaultdict
import xml.etree.ElementTree as ET
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

for k,v in labels.items():
    print(sum(v), len(v), sum(v)/len(v))

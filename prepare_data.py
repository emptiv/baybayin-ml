import os
import numpy as np
from PIL import Image
from bidict import bidict
import csv

ENCODER = bidict({
    'a': 1,
    'o_u': 2, 
    'e_i': 3
})

IMAGE_DIR = './raw_v'
OUTPUT_DIR = './data_v_1'
IMG_SIZE = (50, 50)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

images = []
labels = []
filenames = []  # ← Track filenames here

csv_log_path = os.path.join(OUTPUT_DIR, "label_log.csv")
csv_file = open(csv_log_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["filename", "assigned_label", "label_index"])  # Write header

for filename in os.listdir(IMAGE_DIR):
    if filename.endswith(".jpg"):
        try:
            label = filename.split('.', 1)[0].lower()

            if label not in ENCODER:
                print(f"Skipping: {filename} — unknown label '{label}'")
                continue

            csv_writer.writerow([filename, label, ENCODER[label]])

            img_path = os.path.join(IMAGE_DIR, filename)
            img = Image.open(img_path).convert("L")
            img = img.resize(IMG_SIZE)
            img_arr = np.array(img).astype("float32") / 255.0

            images.append(img_arr)
            labels.append(ENCODER[label])
            filenames.append(filename)  # ← Save filename

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Convert to arrays and save
images = np.array(images).reshape(-1, 50, 50)
labels = np.array(labels)
filenames = np.array(filenames)  # ← Convert to np array

np.save(os.path.join(OUTPUT_DIR, "imgs.npy"), images)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)
np.save(os.path.join(OUTPUT_DIR, "filenames.npy"), filenames)  # ← Save to disk

print(f"Saved {len(images)} images, labels, and filenames to {OUTPUT_DIR}/")

import collections
label_counts = collections.Counter(labels)
for label_int, count in sorted(label_counts.items()):
    print(f"Label {label_int:2} ({ENCODER.inverse[label_int]}): {count} samples")

csv_file.close()
print(f"CSV log saved to {csv_log_path}")

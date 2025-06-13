import os
import numpy as np
from PIL import Image
from bidict import bidict

# Same encoder as your app.py
ENCODER = bidict({
    'a': 1, 
    'e_i' : 2,
    'o_u': 3
})

IMAGE_DIR = './raw_images'
OUTPUT_DIR = './data'
IMG_SIZE = (28, 28)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

images = []
labels = []

for filename in os.listdir(IMAGE_DIR):
    if filename.endswith(".jpg"):
        try:
            label = filename.split('.', 1)[0].lower()

            if label not in ENCODER:
                print(f"Skipping: {filename} — unknown label '{label}'")
                continue

            img_path = os.path.join(IMAGE_DIR, filename)
            img = Image.open(img_path).convert("L")
            img = img.resize(IMG_SIZE)
            img_arr = np.array(img).astype("float32") / 255.0  # normalize

            images.append(img_arr)
            labels.append(ENCODER[label])

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# convert to arrays
images = np.array(images).reshape(-1, 28, 28)  # grayscale
labels = np.array(labels)

# save to disk
np.save(os.path.join(OUTPUT_DIR, "imgs.npy"), images)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)

print(f"Saved {len(images)} images and labels to {OUTPUT_DIR}/")


import collections

label_counts = collections.Counter(labels)
for label_int, count in sorted(label_counts.items()):
    print(f"Label {label_int:2} ({ENCODER.inverse[label_int]}): {count} samples")
import os
import numpy as np
from PIL import Image
from bidict import bidict

# Same encoder as your app.py
ENCODER = bidict({
    'a': 1, 'b': 2, 'ba': 3, 'be_bi': 4, 'bo_bu': 5,
    'd': 6, 'da_ra': 7, 'de_di': 8, 'do_du': 9,
    'e_i': 10, 'g': 11, 'ga': 12, 'ge_gi': 13, 'go_gu': 14,
    'h': 15, 'ha': 16, 'he_hi': 17, 'ho_hu': 18,
    'k': 19, 'ka': 20, 'ke_ki': 21, 'ko_ku': 22,
    'l': 23, 'la': 24, 'le_li': 25, 'lo_lu': 26,
    'm': 27, 'ma': 28, 'me_mi': 29, 'mo_mu': 30,
    'n': 31, 'na': 32, 'ne_ni': 33, 'no_nu': 38,
    'ng': 34, 'nga': 35, 'nge_ngi': 36, 'ngo_ngu': 37,
    'o_u': 39, 'p': 40, 'pa': 41, 'pe_pi': 42, 'po_pu': 43,
    'r': 44, 'ra': 45, 're_ri': 46, 'ro_ru': 47,
    's': 48, 'sa': 49, 'se_si': 50, 'so_su': 51,
    't': 52, 'ta': 53, 'te_ti': 54, 'to_tu': 55,
    'w': 56, 'wa': 57, 'we_wi': 58, 'wo_wu': 59,
    'y': 60, 'ya': 61, 'ye_yi': 62, 'yo_yu': 63
})

IMAGE_DIR = './raw_images'
OUTPUT_DIR = './data'
IMG_SIZE = (50, 50)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

images = []
labels = []

for filename in os.listdir(IMAGE_DIR):
    if filename.endswith(".jpg"):
        try:
            parts = filename.split('.')
            label = parts[0].lower()

            if label not in ENCODER:
                print(f"Skipping: {filename} — unknown label '{label}'")
                continue

            img_path = os.path.join(IMAGE_DIR, filename)
            img = Image.open(img_path).convert("L")
            img = img.resize(IMG_SIZE)
            img_arr = np.array(img).astype("float32") / 255.0  # normalize

            images.append(img_arr)
            labels.append(label)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# convert to arrays
images = np.array(images).reshape(-1, 50, 50)  # grayscale
labels = np.array(labels)

# save to disk
np.save(os.path.join(OUTPUT_DIR, "imgs.npy"), images)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)

print(f"Saved {len(images)} images and labels to {OUTPUT_DIR}/")

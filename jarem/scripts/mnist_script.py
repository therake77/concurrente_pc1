import os
import numpy as np
from PIL import Image
from torchvision.datasets import MNIST  #type: ignore

# --- Configuration ---
DIGIT_A = 0
DIGIT_B = 1
SAMPLES_PER_DIGIT_TRAIN = 500
SAMPLES_PER_DIGIT_TEST = 150
N_PREVIEW_PNGS = 5
SEED = 42
OUTPUT_DIR = "./temp"

os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)

# Load dataset
train_dataset = MNIST(root='./data', train=True, download=True)
test_dataset = MNIST(root='./data', train=False, download=True)


def select_two_digits(dataset, digit_a, digit_b, per_digit):    #type: ignore
    images = dataset.data.numpy()      #type: ignore
    labels = dataset.targets.numpy()   #type: ignore

    idx_a = np.flatnonzero(labels == digit_a) #type: ignore
    idx_b = np.flatnonzero(labels == digit_b) #type: ignore

    rng.shuffle(idx_a)
    rng.shuffle(idx_b)

    idx_a = idx_a[:per_digit]
    idx_b = idx_b[:per_digit]

    if len(idx_a) < per_digit or len(idx_b) < per_digit:
        raise RuntimeError(
            f"Not enough samples: got {len(idx_a)} of digit {digit_a}, "
            f"{len(idx_b)} of digit {digit_b}, wanted {per_digit} each."
        )

    idx = np.concatenate([idx_a, idx_b])
    rng.shuffle(idx)

    raw_images = images[idx] #type: ignore
    # Remap: digit_a -> class 0, digit_b -> class 1
    raw_labels = np.where(labels[idx] == digit_a, 0, 1).astype(np.int64) #type: ignore
    return raw_images, raw_labels #type: ignore


def save_preview_pngs(prefix, raw_images, raw_labels, count): #type: ignore
    for i in range(min(count, len(raw_images))): #type: ignore
        img = Image.fromarray(raw_images[i], mode='L') #type: ignore
        img_resized = img.resize((112, 112), resample=Image.NEAREST)  #type:ignore
        original_digit = DIGIT_A if raw_labels[i] == 0 else DIGIT_B
        img_resized.save(os.path.join(OUTPUT_DIR, f"mnist_{prefix}_sample_{i}_label_{original_digit}.png"))


def export_split(name, raw_images, raw_labels): #type: ignore
    processed_images = (raw_images / 255.0).astype(np.float32) #type: ignore
    processed_images.tofile(os.path.join(OUTPUT_DIR, f"mnist_{name}_images_f32.bin")) #type: ignore

    # One-hot encode: shape (N, 2), matches OutputLayer(dev, 2)
    one_hot = np.eye(2, dtype=np.float32)[raw_labels]
    one_hot.tofile(os.path.join(OUTPUT_DIR, f"mnist_{name}_labels_onehot_f32.bin"))

    print(f"[{name}] saved {len(raw_images)} images " #type: ignore
          f"({np.sum(raw_labels == 0)} of digit {DIGIT_A}, " #type: ignore
          f"{np.sum(raw_labels == 1)} of digit {DIGIT_B})") #type: ignore


train_images, train_labels = select_two_digits(train_dataset, DIGIT_A, DIGIT_B, SAMPLES_PER_DIGIT_TRAIN)
test_images, test_labels = select_two_digits(test_dataset, DIGIT_A, DIGIT_B, SAMPLES_PER_DIGIT_TEST)

# --- 1. Save a few visual PNG copies for human inspection ---
save_preview_pngs("train", train_images, train_labels, N_PREVIEW_PNGS)
save_preview_pngs("test", test_images, test_labels, N_PREVIEW_PNGS)

# --- 2. Preprocess & Export Binary Data for C++ ---
export_split("train", train_images, train_labels)
export_split("test", test_images, test_labels)

print("Done. Classes: 0 ->", DIGIT_A, " 1 ->", DIGIT_B)

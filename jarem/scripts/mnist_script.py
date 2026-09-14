import numpy as np
from PIL import Image
from torchvision.datasets import MNIST  #type: ignore

# Load dataset
dataset = MNIST(root='./data', train=True, download=True)

# Select sample count
num_samples = 5
raw_images = dataset.data[:num_samples].numpy()    # Shape: (5, 28, 28), uint8
raw_labels = dataset.targets[:num_samples].numpy()  # Shape: (5,), int64

# --- 1. Save visual PNG copies for human inspection ---
for i in range(num_samples):
    img = Image.fromarray(raw_images[i], mode='L')  # 'L' = 8-bit grayscale
    # Upscale 4x (112x112) so low-res 28x28 images aren't tiny in image viewers
    img_resized = img.resize((112, 112), resample=Image.NEAREST)    #type:ignore
    img_resized.save(f"mnist_sample_{i}_label_{raw_labels[i]}.png")

# --- 2. Preprocess & Export Binary Data for C++ ---
processed_images = (raw_images / 255.0).astype(np.float32)
processed_images.tofile("mnist_images_f32.bin")
raw_labels.astype(np.int32).tofile("mnist_labels_i32.bin")

print(f"Saved {num_samples} PNG preview files and binary streams successfully.")
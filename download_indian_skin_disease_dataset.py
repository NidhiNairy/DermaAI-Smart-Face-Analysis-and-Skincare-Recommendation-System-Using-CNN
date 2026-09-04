import kagglehub

# Download latest version
path = kagglehub.dataset_download("divypatel2000/indian-skin-disease-dataset")

##print("Path to dataset files:", path)


import os

# Collect all image paths
image_paths = []
for root, dirs, files in os.walk(path):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, f))

# Print total count
print(f"Total images found in dataset: {len(image_paths)}")

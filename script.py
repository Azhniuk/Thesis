import os
import shutil
import random

# Define paths
base_dir = "/media/sofi/Data/Thesis folder/zendo6/VISEM_Tracking_Train_v4/Train/"  # Change this if your base path is different
output_dir = "/media/sofi/Data/Thesis folder/data"  # Output folder for YOLO-formatted data

images_train_dir = os.path.join(output_dir, "images/train")
images_val_dir = os.path.join(output_dir, "images/val")
labels_train_dir = os.path.join(output_dir, "labels/train")
labels_val_dir = os.path.join(output_dir, "labels/val")

# Create necessary directories
for dir_path in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
    os.makedirs(dir_path, exist_ok=True)


image_label_pairs = []
folder_count = 0

for folder in sorted(os.listdir(base_dir)):  # Iterate over numbered folders (11, 12, 13, etc.)
    folder_path = os.path.join(base_dir, folder)
    folder_count += 1
    
    images_path = os.path.join(folder_path, "images")
    labels_path = os.path.join(folder_path, "labels")  # Assuming YOLO format labels are here

    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        continue

    for image_file in os.listdir(images_path):
        if image_file.endswith((".jpg", ".png", ".jpeg")):  # Only process image files
            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(labels_path, label_file)
            image_path = os.path.join(images_path, image_file)

            if os.path.exists(label_path):  # Ensure label exists for the image
                image_label_pairs.append((image_path, label_path))

# Shuffle and split the dataset
random.shuffle(image_label_pairs)
split_index = int(0.8 * len(image_label_pairs))  # 80% training, 20% validation

train_pairs = image_label_pairs[:split_index]
val_pairs = image_label_pairs[split_index:]

# Move files to the correct folders
for img_path, lbl_path in train_pairs:
    shutil.copy(img_path, os.path.join(images_train_dir, os.path.basename(img_path)))
    shutil.copy(lbl_path, os.path.join(labels_train_dir, os.path.basename(lbl_path)))

for img_path, lbl_path in val_pairs:
    shutil.copy(img_path, os.path.join(images_val_dir, os.path.basename(img_path)))
    shutil.copy(lbl_path, os.path.join(labels_val_dir, os.path.basename(lbl_path)))

print(f"✅ Dataset converted successfully! Total images: {len(image_label_pairs)}")
print(f"   Training images: {len(train_pairs)}, Validation images: {len(val_pairs)}")
print(folder_count)
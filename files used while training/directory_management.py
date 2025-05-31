import os

labels_dir = "dataset/labels"  

for split in ["train", "val"]:
    path = os.path.join(labels_dir, split)
    for file in os.listdir(path):
        if file.endswith(".txt"):
            file_path = os.path.join(path, file)
            with open(file_path, "r") as f:
                lines = f.readlines()

            # Keep only class 0
            new_lines = [line for line in lines if line.startswith("0 ")]

            with open(file_path, "w") as f:
                f.writelines(new_lines)

print("Fixed labels")

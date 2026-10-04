import os
from PIL import Image

train_path = r"c:\Users\SAIF-PC\Downloads\archive\Training"
test_path = r"c:\Users\SAIF-PC\Downloads\archive\Testing"

classes = sorted(os.listdir(train_path))
print(f"Classes found ({len(classes)}): {classes}")

total_train = 0
total_test = 0

for c in classes:
    tr_imgs = os.listdir(os.path.join(train_path, c))
    te_imgs = os.listdir(os.path.join(test_path, c))
    total_train += len(tr_imgs)
    total_test += len(te_imgs)
    print(f"  Class '{c:12s}': Train = {len(tr_imgs):4d}, Test = {len(te_imgs):4d}")

print(f"Total Training images: {total_train}")
print(f"Total Testing images: {total_test}")

# Check sample image size
sample_img_path = os.path.join(train_path, classes[0], os.listdir(os.path.join(train_path, classes[0]))[0])
with Image.open(sample_img_path) as img:
    print(f"Sample image mode: {img.mode}, size: {img.size}")

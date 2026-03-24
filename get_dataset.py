import os
import json
import shutil
import zipfile

celebamaskhq_zip_path = "./CelebAMask-HQ.zip"
with zipfile.ZipFile(celebamaskhq_zip_path) as zip_ref:
    print("Extracting zipfile...")
    zip_ref.extractall()
# os.remove(celebamaskhq_zip_path)

with open("./files/celeb_anno.json", "r") as f:
    anno = json.load(f)

dst_dataset_dir = "./datasets/celebamaskhq_identity_dataset"
imgs_dir = os.path.join(dst_dataset_dir, "imgs")
celebahq_img_dir = "./CelebAMask-HQ/CelebA-HQ-img"

os.makedirs(imgs_dir, exist_ok=True)

print("Creating celebamaskhq_identity_dataset...")
for identity in anno["victim_ids"]:
    id_dir = os.path.join(imgs_dir, identity)
    os.makedirs(id_dir, exist_ok=True)

    for img_no in anno["victim_imgs"][identity]:
        src_img_path = os.path.join(celebahq_img_dir, f"{img_no}.jpg")
        shutil.copy(src_img_path, id_dir)

print("Removing extracted files...")
shutil.rmtree("./CelebAMask-HQ")

anno_dir = "./files"
dst_anno_dir = os.path.join(dst_dataset_dir, "annotations")
os.makedirs(dst_anno_dir, exist_ok=True)

for filename in os.listdir(anno_dir):
    src_file_path = os.path.join(anno_dir, filename)
    if os.path.isdir(src_file_path):
        shutil.copytree(src_file_path, os.path.join(dst_dataset_dir, filename), dirs_exist_ok=True)
    else:
        shutil.copy(src_file_path, dst_anno_dir)
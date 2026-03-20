import os
import shutil
import json

celeb_anno_path = "../datasets/celebamaskhq_identity_dataset/annotations/celeb_anno.json"
images_root = "../datasets/celebamaskhq_identity_dataset/imgs"
copy_dest_root = "../datasets/celebahq_all_images/imgs"

os.makedirs(copy_dest_root, exist_ok=True)

with open(celeb_anno_path, 'r') as f:
    celeb_anno = json.load(f)

victim_ids = celeb_anno["victim_ids"]
victim_imgs = celeb_anno["victim_imgs"]
segments = [0, 20, 40, 60, 80]

for seg in segments:
    for id in victim_ids[seg:seg+20]:
        for img_no in victim_imgs[id][8:]:
            # print(img_no)
            # print(int(img_no))
            # exit()
            img_path = os.path.join(images_root, id, f"{img_no}.jpg") 
            copy_dest = os.path.join(copy_dest_root)
            copied_file = os.path.join(copy_dest, f"{img_no}.jpg")
            new_name = os.path.join(copy_dest, f"{int(img_no):06d}.jpg")
            shutil.copy(img_path, copy_dest)
            os.rename(copied_file, new_name)
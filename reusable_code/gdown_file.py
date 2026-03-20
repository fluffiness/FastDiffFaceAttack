import gdown
import zipfile
import os
import argparse
import tarfile

def download_file_from_google_drive(url, output_file):
    gdown.download(url, output_file, quiet=False, fuzzy=True)

file_names = [
    "ffhq256.zip", # 0
    "celebahq256.zip", # 1
    "imagenet-compatible.zip", # 2
    "imagenet100.zip", # 3
    "imagenet1000_clsidx_to_labels.txt", # 4
    "prompt-to-prompt-main.zip", # 5
    "face_assets.zip", # 6
    "celebamaskhq_identity_dataset.zip", # 7
    "CelebAMask-HQ.zip", # 8
    "fiw_dataset.zip", # 9
    "celebamaskhq_identity_dataset_align_256.zip", # 10
    "celebamaskhq_identity_dataset_align.zip", # 11
    "age_gender_race.json", # 12
    "latent_cache.tar.gz", # 13
    "latent_cache512.tar.gz", # 14
    "ffhq_256_ckpt.zip", # 15
    "assets.zip", # 16
    "ckpt_official.tar.gz", # 17
    "all_gen_imgs.tar.gz", # 18
    "diffprot_gen_imgs.tar.gz", # 19
]

urls = {
    "ffhq256.zip": "https://drive.google.com/file/d/1zi6OOC9lkVdr1RhKTtDcyHKfU2psXySA/view?usp=sharing",
    "celebahq256.zip": "https://drive.google.com/file/d/1KJrMX12y1LEFFOZRp34Ywmn2A0coXjqp/view?usp=sharing",
    "imagenet-compatible.zip": "https://drive.google.com/file/d/1oXdAjs0ABA0m1VqqpRKL0MpHwllMQ8Nz/view?usp=sharing",
    "imagenet100.zip": "https://drive.google.com/file/d/1-kn2Cm74PhWKVXkoFQm4-M2XDggEvtwh/view?usp=sharing",
    "imagenet1000_clsidx_to_labels.txt": "https://drive.google.com/file/d/1a90UkAn3qXQaDOTdOCPMrhBObMzXSZ-o/view?usp=sharing",
    "prompt-to-prompt-main.zip": "https://drive.google.com/file/d/1GpO3C6C72APy0U1_99tXo7Hn2dpvHJgf/view?usp=sharing",
    "face_assets.zip": "https://drive.google.com/file/d/1grIVr3i77eT50G-BXM9Jsyxbpue6BTkQ/view?usp=sharing",
    "celebamaskhq_identity_dataset.zip": "https://drive.google.com/file/d/1UaNk8lGA-a7tVbZVF2uGYiisES0OlymP/view?usp=sharing",
    "CelebAMask-HQ.zip": "https://drive.google.com/file/d/1yj98_GGjwrrZiN9jK467_ZWMHM8Ct21D/view?usp=sharing",
    "fiw_dataset.zip": "https://drive.google.com/file/d/1Otap8wjQBYKYWgV7g8k84r6ZVNnQnodU/view?usp=sharing",
    "celebamaskhq_identity_dataset_align_256.zip": "https://drive.google.com/file/d/1mFHZjOxGSMGf1Zz2QddDl5yhsoZ5-sqO/view?usp=sharing",
    "celebamaskhq_identity_dataset_align.zip": "https://drive.google.com/file/d/1KZcCa9yDv_zLGw5vVjVO8ALbeynWMdjB/view?usp=sharing",
    "age_gender_race.json": "https://drive.google.com/file/d/1YK4dXtItY5YmE1Z3hxOaJNY3yOlfn9HW/view?usp=sharing",
    "latent_cache.tar.gz": "https://drive.google.com/file/d/1buN_oeyknst2LtzykubEBW8U_3rlaBvD/view?usp=sharing",
    "latent_cache512.tar.gz": "https://drive.google.com/file/d/1UKErHCeUA4hCdgLHCXjfZkQmUFQ4BGBp/view?usp=sharing",
    "ffhq_256_ckpt.zip": "https://drive.google.com/file/d/1ldEGFNblzCzkFtwdUkBQ0WeQO_FXfxue/view?usp=drive_link",
    "assets.zip": "https://drive.google.com/file/d/19_Y0jR789BGciogjjoGtWNEv-5QBiCB7/view?usp=sharing",
    "ckpt_official.tar.gz": "https://drive.google.com/file/d/1esG7b6ZyO71nPbxERzLD6orBj36Y-0DW/view?usp=sharing",
    "all_gen_imgs.tar.gz": "https://drive.google.com/file/d/15gnRgsJPhrhn-P9g3bo7VPMbeXq1RcKS/view?usp=sharing",
    "diffprot_gen_imgs.tar.gz": "https://drive.google.com/file/d/131SA6zwCi3pKmdW_8lfHTSWKcC9LYej9/view?usp=drive_link",
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download files from Google drive.")
    parser.add_argument("-i", "--file_idxs", nargs="*", type=int, help="A list of indexes of the files to be downloaded. If no index is provided, downloads all files in files_list.")
    parser.add_argument("-o", "--output_dir", type=str, help="The path to the directory to which the downloaded files are saved.", default=".")

    args = parser.parse_args()

    file_idxs = args.file_idxs if args.file_idxs else list(range(len(file_names)))
    print(f"downloading files: {[file_names[idx] for idx in file_idxs]}")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for idx in file_idxs:
        name = file_names[idx]
        url = urls[name]
        output_file_path = output_dir + f"/{name}"
        download_file_from_google_drive(url, output_file_path)
        if ".zip" in name:
            zip_out_dir_name = name[:-4]
            zip_out_dir = output_dir + f"/{zip_out_dir_name}"
            os.makedirs(zip_out_dir, exist_ok=True)
            with zipfile.ZipFile(output_file_path, 'r') as zip_ref:
                zip_ref.extractall(zip_out_dir)
            os.remove(output_file_path)
        if ".tar.gz" in name:
            zip_out_dir_name = name[:-7]
            zip_out_dir = output_dir + f"/{zip_out_dir_name}"
            os.makedirs(zip_out_dir, exist_ok=True)
            with tarfile.open(output_file_path, 'r') as tar:
                tar.extractall(zip_out_dir)
            os.remove(output_file_path)

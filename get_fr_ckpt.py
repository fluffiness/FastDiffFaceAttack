import os
import shutil
import zipfile

assets_zip_file = "./assets.zip"
with zipfile.ZipFile(assets_zip_file, 'r') as zip_ref:
    print("Extracting zipfile...")
    zip_ref.extractall()
# os.remove(assets_zip_file)

assets_model_dir = "./assets/models"
fr_model_dir = "./src/fr_models"

print("Moving fr checkpoints...")
for filename in os.listdir(assets_model_dir):
    if ".pth" in filename:
        pth_file_path = os.path.join(assets_model_dir, filename)
        shutil.copy(pth_file_path, fr_model_dir)

print("Removing extracted files...")
shutil.rmtree("./assets")

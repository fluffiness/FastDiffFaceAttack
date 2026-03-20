import torch
from typing import Optional, Tuple, List, Union, Dict
import numpy as np
import os
import random
from PIL import Image
import json
import argparse
from tqdm import tqdm
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DDIMScheduler

from faceutils.datasets import IDLatentPromptDataset, generate_prompt, AGE_GENDER_RACE_MAP, FaceIDDataset
from faceutils.datetime_logger import DatetimeLogger
from faceutils.utils import ModelEnsemble, load_test_models, SemSegCELoss
from faceutils.timer import Timer

from faceutils.inversions import accelerated_invert, cached_inversion
from faceutils.attention_control import AttentionStore
from faceutils.attention_control_utils import register_attention_control, reset_attention_control, aggregate_attention
from faceutils.utils import embed_prompt, get_noise_pred, diffusion_step, sample, tensor2image, image2tensor

import torch.nn.functional as F

from assets.semseg.model import BiSeNet
from face_latent_attack import attack
from config import get_config, Config

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default="./config.yaml", type=str, help='Path to the default config file.')
parser.add_argument('--guidance_scale', default=2.5, type=float)

args = parser.parse_args()
config = get_config(args)

def seed_torch(seed=42):
    """For reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(config.training.seed)

def create_datasets(
    config: Config,
    logger: DatetimeLogger,
    ext: str="jpg"
) -> Tuple[List[FaceIDDataset], List[FaceIDDataset]]:
# ) -> Tuple[List[FaceLatentDataset], List[FaceLatentDataset]]:
    """
    Create train and test sets for the n=num_train_ids identities with the most images.
    The training sets are chosen as the first train_set_size imgs of each id in annotations.json.
    """
    dataset_path = config.dataset.images_root
    train_set_size = config.dataset.train_set_size
    num_train_ids = config.dataset.num_train_ids
    start_idx = config.dataset.identity_start_idx
    annotations_path = config.dataset.annotations
    res = config.dataset.res

    with open(annotations_path, 'r') as f:
        anno = json.load(f)

    victim_ids = anno["victim_ids"][start_idx: start_idx+num_train_ids]
    img_no_lists = anno["victim_imgs"]

    train_sets = []
    test_sets = []
    for id in victim_ids:
        id_dir = os.path.join(dataset_path, id)
        train_set_files = [img_no + '.' + ext for img_no in img_no_lists[id][0:train_set_size]]
        test_set_files = [img_no + '.' + ext for img_no in img_no_lists[id][train_set_size:]]
        train_set = FaceIDDataset(id_dir, train_set_files, res)
        test_set = FaceIDDataset(id_dir, test_set_files, res)
        train_sets.append(train_set)
        test_sets.append(test_set)
        
    logger.log(f"identities in dataset: {victim_ids}")
    logger.log(f"num_identities: {num_train_ids}")
    logger.log(f"train_img_count: {num_train_ids * train_set_size}")
    logger.log(f"test_img_count: {num_train_ids * (16 - train_set_size)}")

    return train_sets, test_sets      
      

if __name__ == "__main__":
    assert config.training.optim_algo in ['adamw', 'adam', 'pgd']

    logger = DatetimeLogger(config.training.save_dir)
    logger.log(repr(config) + '\n')

    timer = Timer()

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    surrogate_model_names = config.training.surrogate_model_names
    test_model_names = config.training.test_model_names
    victim_model_dir = config.training.victim_model_dir

    surrogate_models = load_test_models(surrogate_model_names, victim_model_dir, device)
    surrogate_models = ModelEnsemble(surrogate_models, device)
    test_models = load_test_models(test_model_names, victim_model_dir, device)
    test_models = ModelEnsemble(test_models, device)

    # create datasets
    trainsets, testsets = create_datasets(config, logger)

    res = config.dataset.res
    target_file_path = config.dataset.target_path
    target_img = Image.open(target_file_path).resize((res, res))
    target_img_tensor = image2tensor(target_img, device)
    target_embedding = test_models(target_img_tensor)

    images_root = config.dataset.images_root
    res = config.dataset.res
    THRES_DICT = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
        'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878), 
        'cosface': (0.144840, 0.241045, 0.312703), 'arcface': (0.144840, 0.241045, 0.312703)}
    
    test_model_names = ['facenet', 'irse50', 'ir152', 'mobile_face']

    attack_success_count = {tmn: 0 for tmn in test_model_names}

    total_image_count = 100 * 8

    for i in range(len(testsets)):

        testset = testsets[i]
        for i in range(len(testset)):
            image = testset[i]
            image = image.unsqueeze(0).to(device)
            # print(image.shape)
            # exit()
            img_filename, img_ext = os.path.splitext(testset.image_filenames[i])
            logger.log(f"{testset.identity}, {img_filename}")
            clean_embedding = test_models(image)

            for test_model_name in test_model_names:
                simi = F.cosine_similarity(target_embedding[test_model_name], clean_embedding[test_model_name])
                thres = THRES_DICT[test_model_name][1]

                if simi > thres:
                    attack_success_count[test_model_name] += 1
    
    asr = {}
    for tmn, asc in attack_success_count.items():
        asr[tmn] = asc / total_image_count

    logger.log(f"Baseline ASR: {asr}")


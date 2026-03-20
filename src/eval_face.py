import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict
import numpy as np
import os
import random
from PIL import Image
import json
import argparse
from tqdm import tqdm
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DDIMScheduler

from faceutils.datasets import IDLatentPromptDataset, generate_prompt, AGE_GENDER_RACE_MAP, GroupLatentPromptDataset
from faceutils.datetime_logger import DatetimeLogger
from faceutils.utils import ModelEnsemble, load_test_models, SemSegCELoss
from faceutils.timer import Timer

from faceutils.inversions import accelerated_invert, cached_inversion
from faceutils.attention_control import AttentionStore
from faceutils.attention_control_utils import register_attention_control, reset_attention_control, aggregate_attention
from faceutils.utils import embed_prompt, get_noise_pred, diffusion_step, sample, tensor2image, image2tensor

from assets.semseg.model import BiSeNet
from face_latent_attack import attack
from config import get_config, Config

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

def create_datasets(
    model,
    config: Config,
    logger: DatetimeLogger,
    ext: str="jpg"
) -> List[IDLatentPromptDataset]:
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

    with open(annotations_path, 'r') as f:
        anno = json.load(f)

    victim_ids = anno["victim_ids"][start_idx: start_idx+num_train_ids]
    img_no_lists = anno["victim_imgs"]

    train_sets = []
    test_sets = []
    for id in victim_ids:
        id_dir = os.path.join(dataset_path, id)
        # train_set_files = [img_no + '.' + ext for img_no in img_no_lists[id][0:train_set_size]]
        test_set_files = [img_no + '.' + ext for img_no in img_no_lists[id][train_set_size:]]
        # train_set = IDLatentPromptDataset(model, config, id_dir, train_set_files)
        test_set = IDLatentPromptDataset(model, config, id_dir, test_set_files)
        # train_sets.append(train_set)
        test_sets.append(test_set)
        
    logger.log(f"identities in dataset: {victim_ids}")
    logger.log(f"num_identities: {num_train_ids}")
    logger.log(f"train_img_count: {num_train_ids * train_set_size}")
    logger.log(f"test_img_count: {num_train_ids * (16 - train_set_size)}")

    return test_sets


def group_datasets(trainsets, testsets, num_id_in_group):
    total_num_ids = len(trainsets)

    grouped_trainsets = []
    grouped_testsets = []
    for i in range(0, total_num_ids, num_id_in_group):
        grouped_trainsets.append(GroupLatentPromptDataset(trainsets[i: i+num_id_in_group]))
        grouped_testsets.append(GroupLatentPromptDataset(testsets[i: i+num_id_in_group]))
    
    return grouped_trainsets, grouped_testsets
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default="./config.yaml", type=str, help='Path to the default config file.')
    parser.add_argument('--guidance_scale', default=2.5, type=float)
    parser.add_argument('--ckpt_base_dir', default="../logs/ckpt_official/mobile_face", type=str)
    parser.add_argument('--test_model_name', default="mobile_face", type=str)
    parser.add_argument('--img_ext', default='.png', type=str)

    # parser.add_argument('--num_train_ids', default=20, type=int)
    # parser.add_argument('--prompt_type', default='age_gender_race', type=str)
    # parser.add_argument('--pairwise_loss_weight', default=0.1, type=float)
    # parser.add_argument('--cross_attn_loss_weight', default=12.0, type=float)
    # parser.add_argument('--ca_jtmo', default=0, type=int)
    # parser.add_argument('--total_update_steps', default=64, type=int)

    args = parser.parse_args()
    config = get_config(args)

    seed_torch(config.training.seed)

    logger = DatetimeLogger(config.training.save_dir, f"eval_{args.ckpt_base_dir.split('/')[-1]}")
    # logger.log(repr(config) + '\n')

    timer = Timer()

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    pretrained_diffusion_path = config.diffusion.pretrained_diffusion_path
    num_inference_steps = config.diffusion.diffusion_steps
    start_step = config.diffusion.start_step
    guidance_scale = config.diffusion.guidance_scale
    do_classifier_free_guidance = (guidance_scale > 0.0)

    # Load a pipeline and set up scheduler
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_diffusion_path).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    surrogate_model_names = config.training.surrogate_model_names
    test_model_names = config.training.test_model_names
    victim_model_dir = config.training.victim_model_dir

    surrogate_models = load_test_models(surrogate_model_names, victim_model_dir, device)
    surrogate_models = ModelEnsemble(surrogate_models, device)
    test_models = load_test_models(test_model_names, victim_model_dir, device)
    test_models = ModelEnsemble(test_models, device)

    # create datasets
    testsets = create_datasets(pipe, config, logger)
    
    ckpt_base_dir = args.ckpt_base_dir

    images_root = config.dataset.images_root
    res = config.dataset.res
    THRES_DICT = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
        'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878), 
        'cosface': (0.144840, 0.241045, 0.312703), 'arcface': (0.144840, 0.241045, 0.312703)}

    test_model_name = args.test_model_name

    logger.log(f"Testing ckpt from {ckpt_base_dir}, testing against {test_model_name}")

    target_file_path = config.dataset.test_path
    target_img = Image.open(target_file_path).resize((res, res))
    target_img = image2tensor(target_img, device)
    target_embedding = test_models(target_img)

    timer.tic()
    total_img_cnt = 0
    asc = 0
    save_dir = os.path.join(logger.log_dir, "imgs")
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(len(testsets)), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):

        testset = testsets[i]
        logger.log(f"Testing on id {testset.identity}")

        id_dir = os.path.join(ckpt_base_dir, testset.identity)

        # detect the latest ckpt file
        max_epoch = 0
        for id_dir_filenames in os.listdir(id_dir):
            if "delta" in id_dir_filenames:
                epoch_no = int(id_dir_filenames[id_dir_filenames.find("epoch")+5: id_dir_filenames.find('.')])
                max_epoch = epoch_no if epoch_no > max_epoch else max_epoch

        ckpt_file = f"delta_epoch{max_epoch}.pth"
        logger.log(f"    Generating adv examples using {testset.identity}/{ckpt_file}")
        ckpt_path = os.path.join(id_dir, ckpt_file)
        delta = torch.load(ckpt_path).to(device)

        success_nos = []
        failure_nos = []
        for i in range(len(testset)):
            image, latent, prompt = testset[i]
            latent = latent.unsqueeze(0).to(device)
            img_no, img_ext = os.path.splitext(testset.image_filenames[i])
            if args.img_ext:
                img_ext = args.img_ext
            logger.log(f"        {testset.identity}/{img_no}{img_ext}")
            adv_latent = latent + delta
            adv_image_tensor = sample(pipe, prompt, adv_latent, guidance_scale, num_inference_steps, start_step, return_tensor=True ,res=res)
            adv_embedding = test_models(adv_image_tensor)
            simi = F.cosine_similarity(target_embedding[test_model_name], adv_embedding[test_model_name])
            thres = THRES_DICT[test_model_name][1]
            image = tensor2image(image.unsqueeze(0))[0]
            adv_image = tensor2image(adv_image_tensor)[0]
            # image.save(os.path.join(save_dir, f"{img_no}_clean{img_ext}"))
            adv_image.save(os.path.join(save_dir, f"{int(img_no):06d}{img_ext}"))

            total_img_cnt += 1
            if simi > thres:
                success_nos.append(img_no)
                asc += 1
            else:
                failure_nos.append(img_no)
            
        logger.log(f"    success: {success_nos}")
        logger.log(f"    failure: {failure_nos}")
    
    gen_time = timer.toc(hms=True)
    logger.log(f"total gen time = {gen_time}")
    logger.log(f"Attack success: {asc} times out of {total_img_cnt}")
    logger.log(f"ASR={100*asc/total_img_cnt:.3f}")

    peak_memory = torch.cuda.max_memory_allocated()
    logger.log(f"Peak memory usage: {peak_memory / (1024 ** 2):.2f} MB")

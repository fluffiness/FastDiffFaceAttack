import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

from typing import Tuple, List, Dict
import random
from PIL import Image
import json
import argparse

import numpy as np
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DDIMScheduler

from faceutils.datasets import IDLatentPromptDataset, generate_prompt, GroupLatentPromptDataset
from faceutils.datetime_logger import DatetimeLogger
from faceutils.utils import ModelEnsemble, load_test_models, SemSegCELoss
from faceutils.timer import Timer

from faceutils.inversions import accelerated_invert, cached_inversion
from faceutils.attention_control import AttentionStore
from faceutils.attention_control_utils import register_attention_control, reset_attention_control, aggregate_attention
from faceutils.utils import embed_prompt, get_noise_pred, diffusion_step, image2latent, monitor_gpu_memory

from faceutils.constants import AGE_GENDER_RACE_MAP

from face_latent_attack import attack
from config import get_config, Config

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default="./config.yaml", type=str, help='Path to the default config file.')
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
    model,
    config: Config,
    logger: DatetimeLogger,
    ext: str="jpg"
) -> Tuple[List[IDLatentPromptDataset], List[IDLatentPromptDataset]]:
    """
    Create train and test sets for the n=num_train_ids according to the annotation file.
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
        train_set_files = [img_no + '.' + ext for img_no in img_no_lists[id][0:train_set_size]]
        test_set_files = [img_no + '.' + ext for img_no in img_no_lists[id][train_set_size:]]
        train_set = IDLatentPromptDataset(model, config, id_dir, train_set_files)
        test_set = IDLatentPromptDataset(model, config, id_dir, test_set_files)
        train_sets.append(train_set)
        test_sets.append(test_set)
        
    logger.log(f"identities in dataset: {victim_ids}")
    logger.log(f"num_identities: {num_train_ids}")
    logger.log(f"train_img_count: {num_train_ids * train_set_size}")
    logger.log(f"test_img_count: {num_train_ids * (16 - train_set_size)}")

    targets = anno["assigned_targets"][start_idx: start_idx+num_train_ids]

    return train_sets, test_sets


def invert_target(model: DiffusionPipeline, config: Config) -> Dict:
    """
    returns target_data = {
        "image": Image.Image,
        "prompt": str,
        "latent": torch.Tensor,
        "att_maps": dict
    }
    """
    res = config.dataset.res
    num_inference_steps = config.diffusion.diffusion_steps
    num_fixed_point_iters = config.diffusion.num_fixed_point_iters
    start_step = config.diffusion.start_step
    guidance_scale = config.diffusion.guidance_scale
    target_file_path = config.dataset.target_path
    auto_prompt = config.dataset.auto_prompt

    target_img = Image.open(target_file_path).resize((res, res))
    target_data = {"image": target_img}

    if auto_prompt:
        with open(config.dataset.attribute_path) as f:
            attributes = json.load(f)
        with open(config.dataset.id_attribute_path) as f:
            id_attribute = json.load(f)
        with open(config.dataset.annotations, 'r') as f:
            anno = json.load(f)  

        gender = id_attribute["target"]["gender"]
        gender_str = AGE_GENDER_RACE_MAP["gender"][gender]
        race = id_attribute["target"]["race"]
        race_str = AGE_GENDER_RACE_MAP["race"][race]
        age = attributes["target"]["age"]
        age_str = AGE_GENDER_RACE_MAP["age"][age]
        prompt = generate_prompt(gender_str, age_str, race_str)
        target_data["prompt"] = prompt
    else:
        prompt = config.dataset.target_prompt
        target_data["prompt"] = prompt

    logger.log(f"Inverting target image with prompt: {target_data['prompt']}")
    target_latent = accelerated_invert(
        model=model,
        start_latents=image2latent(model, target_data["image"]),
        prompt=target_data["prompt"],
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        start_step=start_step,
        num_fixed_point_iters=num_fixed_point_iters,
    )
    target_data["latent"] = target_latent

    ca_loss_weight = config.training.cross_attn_loss_weight
    ca_loss_is_targeted = config.training.ca_loss_is_targeted

    target_att_maps = None
    if ca_loss_weight > 0.0 and ca_loss_is_targeted:
        controller = AttentionStore(res)
        register_attention_control(model, controller)
        target_att_maps = {}

        input_prompt = [prompt] * 2
        prompt_embeds, uncond_embeds = embed_prompt(model, input_prompt)
        context = torch.cat([uncond_embeds, prompt_embeds])

        with torch.no_grad():
            input_latents = torch.cat([target_latent] * 2)
            for i in range(start_step, num_inference_steps):
                t = model.scheduler.timesteps[i]
                noise_pred = get_noise_pred(model, input_latents, t, context, do_classifier_free_guidance, guidance_scale)
                input_latents = diffusion_step(input_latents, noise_pred, model.scheduler, t, num_inference_steps)
                att_map = aggregate_attention(1, controller, res//32, ("up", "down"), True, 1, False)
                target_att_maps[i] = att_map
                controller.reset()
        reset_attention_control(model)
    target_data["att_maps"] = target_att_maps

    return target_data


def group_datasets(trainsets, testsets, num_id_in_group):
    total_num_ids = len(trainsets)

    grouped_trainsets = []
    grouped_testsets = []

    dataset_len = len(trainsets[0])
    num_from_each_set = dataset_len // num_id_in_group
    num_slices = num_id_in_group
    for i in range(0, total_num_ids, num_id_in_group):
        for j in range(0, num_slices):
            grouped_trainsets.append(GroupLatentPromptDataset(trainsets[i: i+num_id_in_group], num_from_each_set, j))
            grouped_testsets.append(GroupLatentPromptDataset(testsets[i: i+num_id_in_group], num_from_each_set, j))
    
    return grouped_trainsets, grouped_testsets


if __name__ == "__main__":
    assert config.training.optim_algo in ['adamw', 'adam', 'sgd', 'pgd', 'mifgsm']

    logger = DatetimeLogger(config.training.save_dir)
    logger.log(repr(config) + '\n')
    timer = Timer()

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    pretrained_diffusion_path = config.diffusion.pretrained_diffusion_path
    num_inference_steps = config.diffusion.diffusion_steps
    start_step = config.diffusion.start_step
    guidance_scale = config.diffusion.guidance_scale
    do_classifier_free_guidance = (guidance_scale > 0.0)

    # Load a pipeline and set up scheduler
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_diffusion_path).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    if config.training.monitor_memory:
        monitor_gpu_memory(logger, "After loading SD")

    all_model_names = config.training.all_model_names
    surrogate_model_names = config.training.surrogate_model_names
    test_model_names = config.training.test_model_names
    victim_model_dir = config.training.victim_model_dir

    fr_models = load_test_models(all_model_names, victim_model_dir, device)
    fr_models = ModelEnsemble(fr_models, device)

    if config.training.monitor_memory:
        monitor_gpu_memory(logger, "After loading FR")

    # create datasets
    num_id_in_group = config.dataset.num_id_in_group
    trainsets, testsets = create_datasets(pipe, config, logger)
    if num_id_in_group > 1:
        trainsets, testsets = group_datasets(trainsets, testsets, num_id_in_group)
    target_data = invert_target(pipe, config)
    logger.log('\n')

    timer.tic()
    delta_dict = attack(
        model=pipe, 
        trainsets=trainsets, 
        testsets=testsets, 
        target_data=target_data, 
        fr_models=fr_models, 
        config=config, 
        logger=logger
    )
    timer.toc()
    logger.log(f"training time: {timer.total()}")
    timer.clear()

    peak_memory = torch.cuda.max_memory_allocated()
    logger.log(f"Peak allocated usage: {peak_memory / (1024 ** 2):.2f} MB")

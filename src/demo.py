import os
from typing import Tuple, List, Dict
import random
from PIL import Image
import json
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DDIMScheduler

from faceutils.datasets import IDLatentPromptDataset, generate_prompt, GroupLatentPromptDataset
from faceutils.datetime_logger import DatetimeLogger
from faceutils.utils import ModelEnsemble, load_test_models, SemSegCELoss
from faceutils.timer import Timer

from faceutils.inversions import accelerated_invert, cached_inversion
from faceutils.attention_control import AttentionStore
from faceutils.attention_control_utils import register_attention_control, reset_attention_control, aggregate_attention
from faceutils.utils import embed_prompt, get_noise_pred, diffusion_step, image2latent, monitor_gpu_memory, sample, image2tensor, tensor2image

from faceutils.constants import AGE_GENDER_RACE_MAP, THRES_DICT

from face_latent_attack import attack
from config import get_config, Config

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default="./config.yaml", type=str, help='Path to the default config file.')
parser.add_argument('--images_root', default="../files/imgs", type=str, help="The directory for the demo personal dataset.")
parser.add_argument('--dataset_name', default="demo", type=str, help="Dataset name for creating the cache directory.")
parser.add_argument('--demo_id', default="509", type=str, help="The identity of the personal dataset used in the demo.")
parser.add_argument('--num_train_ids', default=1, type=int, help="Number of identities to include in the datasets.")
parser.add_argument('--pretrained_ckpt', default="../files/pretrained_perturbations/509/delta_epoch9.pth", type=str, help="Path to the pretrained personal perturbation for id=509")
parser.add_argument('--save_dir', default="../output", type=str, help="Directory to which the perturbation and logs are stored")
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

    target_img = Image.open(target_file_path).resize((res, res))
    target_data = {"image": target_img}

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
    dataset_path = config.dataset.images_root
    train_set_size = config.dataset.train_set_size
    annotations_path = config.dataset.annotations
    demo_id = config.extra.demo_id

    with open(annotations_path, 'r') as f:
        anno = json.load(f)

    victim_ids = [demo_id]
    img_no_list = anno["victim_imgs"][demo_id]

    train_set_files = [img_no + '.' + "jpg" for img_no in img_no_list[0:train_set_size]]
    test_set_files = [img_no + '.' + "jpg" for img_no in img_no_list[train_set_size:]]

    id_dir = os.path.join(dataset_path, demo_id)
    print("Inverting trainset...")
    trainsets = [IDLatentPromptDataset(pipe, config, id_dir, train_set_files)]
    print("Inverting testset...")
    testsets = [IDLatentPromptDataset(pipe, config, id_dir, test_set_files)]

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


    ##################################################################
    #                           Evaluation                           #
    ##################################################################
    ckpt_path = os.path.join(logger.log_dir, config.extra.demo_id, "delta_epoch9.pth")
    img_save_dir = os.path.join(logger.log_dir, "adv_ex")
    os.makedirs(img_save_dir, exist_ok=True)
    delta = torch.load(ckpt_path).to(device)
    testset = testsets[0]

    res = config.dataset.res
    test_model_name = [model_name for model_name in all_model_names if model_name not in surrogate_model_names][0]

    test_file_path = config.dataset.test_path
    test_img = Image.open(test_file_path).resize((res, res))
    test_img = image2tensor(test_img, device)
    test_emb = fr_models(test_img)

    timer.tic()

    total_img_cnt = 0
    asc = 0
    success_nos = []
    failure_nos = []
    for i in range(len(testset)):
        image, latent, prompt = testset[i]
        latent = latent.unsqueeze(0).to(device)
        img_no, img_ext = os.path.splitext(testset.image_filenames[i])
        print(f"        {testset.identity}/{img_no}{img_ext}")
        adv_latent = latent + delta
        adv_image_tensor = sample(pipe, prompt, adv_latent, guidance_scale, num_inference_steps, start_step, return_tensor=True ,res=res)
        adv_embedding = fr_models(adv_image_tensor)
        simi = F.cosine_similarity(test_emb[test_model_name], adv_embedding[test_model_name])
        thres = THRES_DICT[test_model_name][1]
        image = tensor2image(image.unsqueeze(0))[0]
        adv_image = tensor2image(adv_image_tensor)[0]
        adv_image.save(os.path.join(img_save_dir, f"{int(img_no):06d}{img_ext}"))

        total_img_cnt += 1
        if simi > thres:
            success_nos.append(img_no)
            asc += 1
        else:
            failure_nos.append(img_no)

    print("Evaluation results:")
    print(f"    success: {success_nos}")
    print(f"    failure: {failure_nos}")
    
    gen_time = timer.toc(hms=True)
    print(f"    Generation time = {gen_time}, excluding inversion time.")
    print(f"    Attack success: {asc} times out of {total_img_cnt}")
    print(f"    ASR={100*asc/total_img_cnt:.3f}")



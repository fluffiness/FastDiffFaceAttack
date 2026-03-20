# Import modules
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler, DiffusionPipeline
from typing import List, Union, Callable
import numpy as np
import os
from config import Config

from faceutils.utils import image2latent, embed_prompt, get_noise_pred, sample


def params2str(params: dict, exclude: list = []):
    parts = []
    for k, v in sorted(params.items()):
        if k not in exclude:
            key_initials_list = [word[0] for word in k.split('_')]
            k_init = ''.join(key_initials_list)
            safe_val = str(v).replace('/', '_')
            safe_val = str(v).replace(' ', '_')
            parts.append(f"{k_init}={safe_val}")
    return '-'.join(parts)


def cached_inversion(
    inversion_fn: Callable, 
    config: Config,
    model: DiffusionPipeline,
):
    """
    Wrapper function that addes a caching an retrieving layer on top of inversion functions.
    The caching directory is structured:
    cache_dataset_dir
                ├──────── dataset_dir1
                |               ├──────── inversion_param_specific_dir1
                |               |                               ├──────── {img_no1}.pth
                |               |                               ├──────── {img_no2}.pth
                |               |                               ⋮
                |               |
                |               ├──────── inversion_param_specific_dir2
                |               ⋮
                |
                ├──────── dataset_dir2
                ⋮
    """
    cache_dir = config.dataset.cache_dir
    dataset_name = config.dataset.dataset_name
    res = config.dataset.res
    prompt_type = config.dataset.prompt_type
    cache_dataset_dir = os.path.join(cache_dir, dataset_name)
    os.makedirs(cache_dataset_dir, exist_ok=True)

    def cache_wrapper(img_path, **params):
        save_dir = os.path.join(cache_dataset_dir, params2str(params, "prompt")+f'-res={res}-ptype={prompt_type}')
        os.makedirs(save_dir, exist_ok=True)

        filename_without_ext = os.path.splitext(os.path.basename(img_path))[0]
        latent_file_name = filename_without_ext + ".pth"
        latent_path = os.path.join(save_dir, latent_file_name)

        if os.path.exists(latent_path):
            return torch.load(latent_path)

        image = Image.open(img_path).resize((res, res))
        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1
        start_latent = image2latent(model, image)
        latent = inversion_fn(model, start_latent, **params)
        torch.save(latent, latent_path)
        return latent
    
    return cache_wrapper


def inversion_step(
    latents: torch.Tensor, 
    noise_pred: torch.Tensor, 
    scheduler,
    t: int,
    num_inference_steps: int,
) -> torch.Tensor:
    
    num_train_timesteps = scheduler.config.num_train_timesteps
    current_t = max(0, t - (num_train_timesteps // num_inference_steps))  # t
    next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
    alpha_t = scheduler.alphas_cumprod[current_t]
    alpha_t_next = scheduler.alphas_cumprod[next_t]

    # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
    predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

    return predicted_x0 * alpha_t_next.sqrt() + (1 - alpha_t_next).sqrt() * noise_pred


## DDIM Inversion
@torch.no_grad()
def ddim_invert(
    model: DiffusionPipeline,
    start_latents: torch.Tensor,
    prompt: Union[str, List[str]],
    guidance_scale: float = 3.5,
    num_inference_steps: int = 20,
    start_step: int = 10,
    return_intermediate: bool = False,
    do_classifier_free_guidance: bool = True,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    
    device = model.device

    prompt_embeds, uncond_embeds = embed_prompt(model, prompt)
    context = torch.cat([uncond_embeds, prompt_embeds])

    latents = start_latents
    intermediate_latents = [start_latents]

    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = reversed(model.scheduler.timesteps)

    # print("DDIM inversion...")
    for i in range(1, num_inference_steps-start_step):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue
        t = timesteps[i]
        noise_pred = get_noise_pred(model, latents, t, context, do_classifier_free_guidance, guidance_scale)
        latents = inversion_step(latents, noise_pred, model.scheduler, t.item(), num_inference_steps)

        # Store
        intermediate_latents.append(latents)

    return intermediate_latents if return_intermediate else intermediate_latents[-1]


## AIDI
@torch.no_grad()
def accelerated_invert(
    model: DiffusionPipeline,
    start_latents: torch.Tensor,
    prompt: Union[str, List[str]],
    guidance_scale: float = 3.5,
    num_inference_steps: int = 20,
    start_step: int = 10,
    num_fixed_point_iters: int = 20,
    return_intermediate: bool = False,
    do_classifier_free_guidance: bool = True,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    If start_latents is in a batch, prompt must be a list of length==batch_size
    """
    
    device = model.device

    prompt_embeds, uncond_embeds = embed_prompt(model, prompt)
    context = torch.cat([uncond_embeds, prompt_embeds])

    latents = start_latents.clone()
    intermediate_latents = [start_latents]

    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = reversed(model.scheduler.timesteps)

    # print("AIDI inversion...")
    latents_current = latents
    best_iters = []
    for i in range(1, num_inference_steps-start_step):
    # for i in range(1, num_inference_steps):

        t = timesteps[i]
        
        num_train_timesteps = model.scheduler.config.num_train_timesteps
        current_t = max(0, t.item() - (num_train_timesteps // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = model.scheduler.alphas_cumprod[current_t]
        alpha_t_next = model.scheduler.alphas_cumprod[next_t]

        a_t = (alpha_t_next / alpha_t).sqrt()
        b_t = (1 - alpha_t_next).sqrt() - ((1 - alpha_t) * alpha_t_next / alpha_t).sqrt()

        def fixed_point_func(latent_i):
            noise_pred = get_noise_pred(model, latent_i, t, context, do_classifier_free_guidance, guidance_scale)
            return a_t * latents_current + b_t * noise_pred

        # efficient fixed-point iteration due to the author of the AIDI paper
        latents_i1 = latents_current
        latents_i2 = fixed_point_func(latents_i1)
        func_latents_i1 = latents_i2
        func_latents_i2 = fixed_point_func(latents_i2)
        best_iter = 0
        latents_best = latents_i2
        min_diff = torch.linalg.norm(latents_best - func_latents_i2).item()
        for i in range(num_fixed_point_iters):
            latents_i1, latents_i2 = latents_i2, 0.5 * (func_latents_i1 + func_latents_i2)
            func_latents_i1, func_latents_i2 = func_latents_i2, fixed_point_func(latents_i2)
            diff = torch.linalg.norm(latents_i2 - func_latents_i2).item()
            if diff < min_diff:
                min_diff = diff
                latents_best = latents_i2
                best_iter = i+1
        best_iters.append(best_iter)

        latents_next = latents_best

        # Store
        intermediate_latents.append(latents_next)
        latents_current = latents_next
    
    # print("best_iters: ", best_iters)

    return intermediate_latents if return_intermediate else intermediate_latents[-1]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")

    # Load a pipeline and set up scheduler
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base").to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    img_paths = ["./img/input_img.jpeg", "./img/input_img2.png", "./img/input_img4.jpg"]
    input_image_prompt = ["puupy", "hummingbird", "the face of a woman"] # puppy, hummingbird, scabbard
    res = 256
    input_images = [Image.open(img_path).resize((res, res)) for img_path in img_paths]
    latents = torch.cat([image2latent(pipe, img, sample=False) for img in input_images])
    print("latents.shape: ", latents.shape)

    num_inference_steps = 20
    num_fixed_point_iters = 15
    start_step = 10

    guidance_scales = [2.5]
    for guidance_scale in guidance_scales:
        print("Inverting...")
        # inverted_latents = ddim_invert(pipe, latents, input_image_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        inverted_latents = accelerated_invert(
            pipe, 
            latents, 
            input_image_prompt, 
            num_inference_steps=num_inference_steps, 
            num_fixed_point_iters = num_fixed_point_iters,
            guidance_scale=guidance_scale, 
        )
        # delta = torch.load("/tmp2/r10922106/diff_attack/src/logs/11-28_17:28/161/iter1/delta_iter1.pt")
        # start_latent = inverted_latents[-1-start_step]
        start_latent = inverted_latents
        print("start_latent.shape: ", start_latent.shape)
        print("Sampling...")
        resampled_imgs = sample(
            pipe, 
            input_image_prompt,  
            start_latent, 
            guidance_scale=guidance_scale, 
            num_inference_steps=num_inference_steps,
            start_step=start_step,
            res=res,
        )

        resampled_img1 = resampled_imgs[0]
        resampled_img2 = resampled_imgs[1]
        resampled_img4 = resampled_imgs[2]
        save_path1 = f"./img/resampled_img1_w={guidance_scale}_start={start_step}_batch_{num_fixed_point_iters}iters.jpg"
        save_path2 = f"./img/resampled_img2_w={guidance_scale}_start={start_step}_batch_{num_fixed_point_iters}iters.jpg"
        save_path4 = f"./img/resampled_img4_w={guidance_scale}_start={start_step}_batch_{num_fixed_point_iters}iters.jpg"
        resampled_img1.save(save_path1)
        resampled_img2.save(save_path2)
        resampled_img4.save(save_path4)

        # resampled_img = resampled_imgs[0]
        # save_path = f"./img/resampled_img3_w={guidance_scale}_start={start_step}_batch_{num_fixed_point_iters}iters.jpg"
        # resampled_img.save(save_path)

        print(f"saved resampled images to:\n\t{save_path1}\n\t{save_path2}\n\t{save_path4}")
        # resampled_img_official.save(f"./img/resampled_img_official_w={guidance_scale}.jpeg")

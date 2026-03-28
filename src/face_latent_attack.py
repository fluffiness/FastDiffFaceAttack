import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict
import numpy as np
import os
from PIL import Image
from diffusers import DiffusionPipeline
from torch.utils.data import DataLoader
import torchvision.transforms.functional as Fv
import random
import re
from tqdm import tqdm

from faceutils.utils import image2tensor, tensor2image, image2latent, latent2image, embed_prompt, get_noise_pred, diffusion_step, sample
from faceutils.utils import ModelEnsemble, cos_dist_loss, neighborhood_struct_loss, get_mean_asr, monitor_gpu_memory, latent_diff_boundary
from faceutils.similarity_metrics import calculate_ssim, calculate_lpips
from faceutils.attention_loss import cross_attention_loss, attn_structural_loss, targeted_cross_attention_loss
from faceutils.timer import Timer, sec2hms
from faceutils.constants import THRES_DICT, KEY_WORDS

from faceutils.optimizers import AdamW, PGD, SGD, MIFGSM
from faceutils.attention_control_utils import register_attention_control, reset_attention_control, aggregate_attention
from faceutils.attention_control import StructureLossAttentionStore, AttentionStore
from faceutils.datetime_logger import DatetimeLogger
from faceutils.datasets import IDLatentPromptDataset, GroupLatentPromptDataset

from config import Config


"""
    The latents are duplicated and organized as follows:
        For the attention losses: latents = torch.cat([latents, adv_latents])
        For cfg: latents = torch.cat([latents * 2])
    During the noise prediction process, the input to the unit is 4 sets of the latents
    The first 2 are for the unconditional prediction, and the last 2 are for conditional
                  embeddings       latents
               ┌──────────────┬──────────────┐
     batch 1   │     null     │    clean     │
               ├──────────────┼──────────────┤
     batch 2   │     null     │     adv      │
               ├──────────────┼──────────────┤
     batch 3   │     text     │    clean     │
               ├──────────────┼──────────────┤
     batch 4   │     text     │     adv      │
               └──────────────┴──────────────┘
    In the AttentionControl modules, batch 1 and 2 are not stored since we only care about 
    the cross-attention with text embeddings.
    For self-attentions control, we calculate the MSE-loss of the self-attention maps
    between batch 3 and 4
"""


@torch.enable_grad()
def identity_attack_memory_efficient(
    model: DiffusionPipeline,
    trainset: Union[IDLatentPromptDataset, GroupLatentPromptDataset],
    target_data: dict,
    test_img: torch.Tensor,
    fr_models: ModelEnsemble,
    ckpt_dir: str,
    config: Config,
    logger: DatetimeLogger,
) -> torch.Tensor:
    device = model.device
    num_inference_steps = config.diffusion.diffusion_steps
    start_step = config.diffusion.start_step
    guidance_scale = config.diffusion.guidance_scale
    do_classifier_free_guidance = (guidance_scale > 0.0)

    surrogate_model_names = config.training.surrogate_model_names

    # clip_min = -1e6
    # clip_max = 1e6

    res = config.dataset.res
    prompt_type = config.dataset.prompt_type

    batch_size = config.training.batch_size
    lr = config.training.lr
    warm_up_steps = config.training.warm_up_steps
    pgd_radius = config.training.pgd_radius
    weight_decay = config.training.weight_decay
    momentum = config.training.momentum
    lam_id = config.training.attack_loss_weight
    lam_ca = config.training.cross_attn_loss_weight
    lam_ca_reg = config.training.cross_attn_reg_weight
    lam_sa = config.training.self_attn_loss_weight
    n_epoch = config.training.total_update_steps // (len(trainset) // batch_size)
    # ca_loss_is_targeted = config.training.ca_loss_is_targeted

    generator = torch.Generator()
    generator.manual_seed(config.training.seed)

    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        generator=generator,
    )

    # train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    target_img = target_data["image"]
    target_img = image2tensor(target_img, device)
    target_embedding = fr_models(target_img, surrogate_model_names)

    controller = AttentionStore(res=res)

    delta = torch.zeros(1, 4, res//8, res//8).to(device)
    if config.training.optim_algo == 'adamw':
        optimizer = AdamW(delta, lr=lr, weight_decay=weight_decay, warm_up_steps=warm_up_steps)
    elif config.training.optim_algo == 'adam':
        optimizer = AdamW(delta, lr=lr, weight_decay=0.0, warm_up_steps=warm_up_steps)
    elif config.training.optim_algo == 'sgd':
        optimizer = SGD(delta, lr=lr, weight_decay=weight_decay, warm_up_steps=warm_up_steps)
    elif config.training.optim_algo == 'pgd':
        optimizer = PGD(delta, lr=lr, radius=pgd_radius, warm_up_iters=warm_up_steps)
    elif config.training.optim_algo == 'mifgsm':
        optimizer = MIFGSM(delta, lr=lr, radius=pgd_radius, warm_up_iters=warm_up_steps, momentum=momentum)
    else:
        raise ValueError("Invalid optimizer type")

    for e in range(n_epoch):
        register_attention_control(model, controller)
        for it, batch in enumerate(train_loader):
            images: torch.Tensor
            batch_diff_latents: torch.Tensor
            images, batch_diff_latents, batch_prompt = batch
            images = images.to(device)
            batch_diff_latents = batch_diff_latents.to(device)
            batch_diff_latents.requires_grad_(False)

            prompt = batch_prompt * 2 # since we must calculate the clean and adv latents together
            prompt_embeds, uncond_embeds = embed_prompt(model, prompt)
            context = torch.cat([uncond_embeds, prompt_embeds])
            context.requires_grad_(False)

            target_prompt = target_data["prompt"]
            target_prompt = [target_prompt] * batch_size * 2
            target_prompt_embeds, uncond_embeds = embed_prompt(model, target_prompt)
            target_context = torch.cat([uncond_embeds, target_prompt_embeds])
            target_context.requires_grad_(False)

            input_latents = torch.cat([batch_diff_latents, batch_diff_latents + delta]).detach()
            if config.diffusion.start_step == 0:
                input_latents = latent_diff_boundary(input_latents)
            input_latents.requires_grad_(False)

            # 1. initial dinoising loop to pre-calculate the latents of each timestep
            intermediate_latents = [input_latents]
            with torch.no_grad():
                latents = input_latents
                for i in range(start_step, num_inference_steps):
                    t = model.scheduler.timesteps[i]
                    noise_pred = get_noise_pred(model, latents, t, context, do_classifier_free_guidance, guidance_scale)
                    latents = diffusion_step(latents, noise_pred, model.scheduler, t, num_inference_steps)
                    intermediate_latents.append(latents)
                controller.reset()
            
            if config.training.monitor_memory:
                monitor_gpu_memory(logger, "After itermediate latents precalc, before adv_loss calc")

            # 2. calculate losses on adv_latents and adv_images
            # calculate adv_loss loss from adv_latent z0
            adv_vae_latents = intermediate_latents[-1][batch_size:] # discard the clean latents
            adv_vae_latents.requires_grad_(True)
            adv_images = latent2image(model, adv_vae_latents, return_tensor=True) # [-1, 1]

            adv_embeddings = fr_models(adv_images, surrogate_model_names)
            # print(list(adv_embeddings.keys()))
            adv_loss = lam_id * cos_dist_loss(adv_embeddings, target_embedding)

            total_loss = 0.0 + adv_loss

            if config.training.monitor_memory:
                monitor_gpu_memory(logger, "Before adv grad")
            
            # 3. initialize the gradient as dL/dz0
            grad = torch.autograd.grad(total_loss, adv_vae_latents, retain_graph=False)[0].detach()
            torch.cuda.empty_cache()
            del adv_embeddings, adv_images, adv_vae_latents

            total_loss = total_loss.detach()

            sa_loss = torch.tensor(0.0).to(device) # self-attention loss
            ca_reg_loss = torch.tensor(0.0).to(device)
            ca_loss = torch.tensor(0.0).to(device) # cross-attention loss

            if config.training.monitor_memory:
                monitor_gpu_memory(logger, "After adv grad, before mem eff grad loop")

            # 4. memory efficient gradient calculation
            for i in reversed(list(range(start_step, num_inference_steps))):
                t = model.scheduler.timesteps[i]

                # retreive the pre-calculated latents and set the adv half to requires_grad = True
                latent_idx = i - start_step
                latents_pre_calc = intermediate_latents[latent_idx]
                clean_latents_pre_calc, adv_latents_pre_calc = torch.chunk(latents_pre_calc, 2)
                adv_latents_pre_calc.requires_grad_(True)
                latents_pre_calc = torch.cat([clean_latents_pre_calc, adv_latents_pre_calc])

                # 4-1. calculate self-attention loss and calculate cross-attention regularization loss
                noise_pred_from_pre_calc = get_noise_pred(model, latents_pre_calc, t, context, do_classifier_free_guidance, guidance_scale)
                next_latents = diffusion_step(latents_pre_calc, noise_pred_from_pre_calc, model.scheduler, t, num_inference_steps)

                if config.training.monitor_memory:
                    monitor_gpu_memory(logger, "After denoising step for sa and ca_reg")

                step_sa_loss = torch.tensor(0.0).to(device)
                step_ca_reg_loss = torch.tensor(0.0).to(device)
                if lam_ca_reg > 0 or lam_sa > 0:
                    step_ca_reg_loss, step_sa_loss = attn_structural_loss(controller, batch_prompt, model.tokenizer, lam_ca_reg, lam_sa, words=config.training.words, device=device)
                    # print(f"step_sa_loss: {step_sa_loss.item()}, step_sa_loss_alt: {step_sa_loss_alt.item()}")

                step_loss_for_grad = torch.sum(next_latents[batch_size:] * grad) + step_sa_loss + step_ca_reg_loss # next_grad = d/dx(f_r) * grad + d/dx(step_loss)
                if config.training.monitor_memory:
                    monitor_gpu_memory(logger, "Right before sa and ca_reg grad")
                grad = torch.autograd.grad(step_loss_for_grad, adv_latents_pre_calc, retain_graph=False)[0].detach()

                if config.training.monitor_memory:
                    monitor_gpu_memory(logger, "Right after sa and ca_reg grad")

                total_loss += step_sa_loss.detach()
                sa_loss += step_sa_loss.detach()
                ca_reg_loss += step_ca_reg_loss.detach()

                controller.reset()
                del noise_pred_from_pre_calc, next_latents, step_sa_loss, step_ca_reg_loss, step_loss_for_grad
                torch.cuda.empty_cache()

                if config.training.monitor_memory:
                    monitor_gpu_memory(logger, "After empty cache and del for sa and ca_reg")

                # 4-2. calculate cross-attention loss
                step_ca_loss = torch.tensor(0.0)
                if lam_ca > 0.0:
                    noise_pred_from_pre_calc_ca = get_noise_pred(model, latents_pre_calc, t, target_context, do_classifier_free_guidance, guidance_scale)
                    next_latents_ca = diffusion_step(latents_pre_calc, noise_pred_from_pre_calc_ca, model.scheduler, t, num_inference_steps)

                    step_ca_loss = targeted_cross_attention_loss(
                        controller = controller,
                        batch_size = batch_size,
                        target_prompt = target_data["prompt"], 
                        target_att_map = target_data["att_maps"][i], 
                        tokenizer = model.tokenizer, 
                        config = config
                    )
                    step_ca_loss = lam_ca * step_ca_loss

                    if config.training.monitor_memory:
                        monitor_gpu_memory(logger, "Right before ca grad")
                    grad_ca = torch.autograd.grad(step_ca_loss, adv_latents_pre_calc, retain_graph=False)[0].detach()
                    if config.training.monitor_memory:
                        monitor_gpu_memory(logger, "Right after ca grad")

                    total_loss += step_ca_loss.detach()
                    ca_loss += step_ca_loss.detach()

                    del noise_pred_from_pre_calc_ca, next_latents_ca, step_ca_loss
                    torch.cuda.empty_cache()
                    if config.training.monitor_memory:
                        monitor_gpu_memory(logger, "After empty cache and del for ca")


                    grad = grad + grad_ca
                    controller.reset()

            grad = grad.sum(dim=0, keepdim=True)
            optimizer(grad)
            logger.log(f"epoch{e}, iteration{it}")
            logger.log((
                f"    adv_loss: {adv_loss.item():.3f}, "
                f"cross_attention_loss: {ca_loss.item():.3f}, "
                f"cross_attention_regularizerion_loss: {ca_reg_loss.item():.3f}, "
                f"self_attention_loss: {sa_loss.item():.3f}, "
                f"total_loss: {total_loss.item():.3f}"
            ), out=False)
            logger.log(f"    delta norm, l2: {delta.norm()}, linf: {delta.abs().max()}", out=False)
        
        reset_attention_control(model)

        ckpt_file = os.path.join(ckpt_dir, f"delta_epoch{e}.pth")
        torch.save(delta, ckpt_file)
        asr, ssim, lpips = eval(
            model = model, 
            testset = trainset, 
            delta = delta, 
            target_img = test_img, 
            fr_models = fr_models, 
            save_sample_img = False, 
            ckpt_dir = ckpt_dir, 
            calculate_sim = True, 
            config = config, 
            logger = logger
        )
        logger.log(f"epoch{e} trainset asr on id {trainset.identity}: {asr}", out=False)
        logger.log(f"trainset perceptual sim: ssim={ssim:.6f}, lpips={lpips:.6f}", out=False)
        logger.log('', out=False)
        peak_memory = torch.cuda.max_memory_allocated()
        logger.log(f"Peak memory usage: {peak_memory / (1024 ** 2):.2f} MB\n", out=False)

    return delta


@torch.enable_grad()
def identity_attack_regular(
    model: DiffusionPipeline,
    trainset: Union[IDLatentPromptDataset, GroupLatentPromptDataset],
    target_data: dict,
    test_img: torch.Tensor,
    fr_models: ModelEnsemble,
    ckpt_dir: str,
    config: Config,
    logger: DatetimeLogger,
) -> torch.Tensor:
    device = model.device
    num_inference_steps = config.diffusion.diffusion_steps
    start_step = config.diffusion.start_step
    guidance_scale = config.diffusion.guidance_scale
    do_classifier_free_guidance = (guidance_scale > 0.0)

    surrogate_model_names = config.training.surrogate_model_names

    # clip_min = -1e6
    # clip_max = 1e6

    res = config.dataset.res
    prompt_type = config.dataset.prompt_type

    batch_size = config.training.batch_size
    lr = config.training.lr
    warm_up_steps = config.training.warm_up_steps
    pgd_radius = config.training.pgd_radius
    weight_decay = config.training.weight_decay
    momentum = config.training.momentum
    lam_id = config.training.attack_loss_weight
    lam_ca = config.training.cross_attn_loss_weight
    lam_ca_reg = config.training.cross_attn_reg_weight
    lam_sa = config.training.self_attn_loss_weight
    n_epoch = config.training.total_update_steps // (len(trainset) // batch_size)
    # ca_loss_is_targeted = config.training.ca_loss_is_targeted

    generator = torch.Generator()
    generator.manual_seed(config.training.seed)

    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        generator=generator,
    )

    # train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    target_img = target_data["image"]
    target_img = image2tensor(target_img, device)
    target_embedding = fr_models(target_img, surrogate_model_names)

    controller = AttentionStore(res=res)

    delta = torch.zeros(1, 4, res//8, res//8).to(device)
    delta.requires_grad_(True)
    if config.training.optim_algo == 'adamw':
        optimizer = AdamW(delta, lr=lr, weight_decay=weight_decay, warm_up_steps=warm_up_steps)
    elif config.training.optim_algo == 'adam':
        optimizer = AdamW(delta, lr=lr, weight_decay=0.0, warm_up_steps=warm_up_steps)
    elif config.training.optim_algo == 'sgd':
        optimizer = SGD(delta, lr=lr, weight_decay=weight_decay, warm_up_steps=warm_up_steps)
    elif config.training.optim_algo == 'pgd':
        optimizer = PGD(delta, lr=lr, radius=pgd_radius, warm_up_iters=warm_up_steps)
    elif config.training.optim_algo == 'mifgsm':
        optimizer = PGD(delta, lr=lr, radius=pgd_radius, warm_up_iters=warm_up_steps)

    for e in range(n_epoch):
        register_attention_control(model, controller)
        for it, batch in enumerate(train_loader):
            images: torch.Tensor
            batch_diff_latents: torch.Tensor
            images, batch_diff_latents, batch_prompt = batch
            images = images.to(device)
            batch_diff_latents = batch_diff_latents.to(device)
            batch_diff_latents

            prompt = batch_prompt * 2 # since we must calculate the clean and adv latents together
            prompt_embeds, uncond_embeds = embed_prompt(model, prompt)
            context = torch.cat([uncond_embeds, prompt_embeds])
            context.requires_grad_(False)

            target_prompt = target_data["prompt"]
            target_prompt = [target_prompt] * batch_size * 2
            target_prompt_embeds, uncond_embeds = embed_prompt(model, target_prompt)
            target_context = torch.cat([uncond_embeds, target_prompt_embeds])
            target_context.requires_grad_(False)

            adv_diff_latents = batch_diff_latents + delta
            input_latents = torch.cat([batch_diff_latents, adv_diff_latents])

            intermediate_latents = [input_latents]
            total_loss = 0.0
            sa_loss = torch.tensor(0.0).to(device) # self-attention loss
            ca_reg_loss = torch.tensor(0.0).to(device)
            ca_loss = torch.tensor(0.0).to(device) # cross-attention loss
            latents = input_latents

            if config.training.monitor_memory:
                monitor_gpu_memory(logger, "Before denoising loop")
            for i in range(start_step, num_inference_steps):
                t = model.scheduler.timesteps[i]
                noise_pred = get_noise_pred(model, latents, t, context, do_classifier_free_guidance, guidance_scale)
                next_latents = diffusion_step(latents, noise_pred, model.scheduler, t, num_inference_steps)
                intermediate_latents.append(next_latents)

                step_sa_loss = torch.tensor(0.0).to(device)
                step_ca_reg_loss = torch.tensor(0.0).to(device)
                if lam_ca_reg > 0 or lam_sa > 0:
                    step_ca_reg_loss, step_sa_loss = attn_structural_loss(controller, batch_prompt, model.tokenizer, lam_ca_reg, lam_sa, words=config.training.words, device=device)
                
                controller.reset()

                step_ca_loss = torch.tensor(0.0)
                if lam_ca > 0.0:
                    noise_pred_ca = get_noise_pred(model, latents, t, target_context, do_classifier_free_guidance, guidance_scale)
                    next_latents_ca = diffusion_step(latents, noise_pred_ca, model.scheduler, t, num_inference_steps)

                    step_ca_loss = targeted_cross_attention_loss(
                        controller = controller,
                        batch_size = batch_size,
                        target_prompt = target_data["prompt"], 
                        target_att_map = target_data["att_maps"][i], 
                        tokenizer = model.tokenizer, 
                        config = config
                    )
                    step_ca_loss = lam_ca * step_ca_loss
                    controller.reset()
                
                sa_loss += step_sa_loss
                ca_reg_loss += step_ca_reg_loss
                ca_loss += step_ca_loss
            if config.training.monitor_memory:
                monitor_gpu_memory(logger, "After denoising loop")

            # 2. calculate losses on adv_latents and adv_images
            # calculate adv_loss loss from adv_latent z0
            adv_vae_latents = intermediate_latents[-1][batch_size:] # discard the clean latents
            adv_images = latent2image(model, adv_vae_latents, return_tensor=True) # [-1, 1]

            adv_embeddings = fr_models(adv_images, surrogate_model_names)
            adv_loss = lam_id * cos_dist_loss(adv_embeddings, target_embedding)

            total_loss = adv_loss + sa_loss + ca_reg_loss + ca_loss
            if config.training.monitor_memory:
                monitor_gpu_memory(logger, "Right before grad denoising loop")
            grad = torch.autograd.grad(total_loss, delta, retain_graph=False)[0].detach()
            total_loss = total_loss.detach()

            if config.training.monitor_memory:
                monitor_gpu_memory(logger, "Right after grad denoising loop")

            sa_loss = torch.tensor(0.0).to(device) # self-attention loss
            ca_reg_loss = torch.tensor(0.0).to(device)
            ca_loss = torch.tensor(0.0).to(device) # cross-attention loss

            optimizer(grad)
            logger.log(f"epoch{e}, iteration{it}")
            logger.log((
                f"    adv_loss: {adv_loss.item():.3f}, "
                f"cross_attention_loss: {ca_loss.item():.3f}, "
                f"cross_attention_regularizerion_loss: {ca_reg_loss.item():.3f}, "
                f"self_attention_loss: {sa_loss.item():.3f}, "
                f"total_loss: {total_loss.item():.3f}"
            ), out=False)
            logger.log(f"    delta norm, l2: {delta.norm()}, linf: {delta.abs().max()}", out=False)
        
        reset_attention_control(model)

        ckpt_file = os.path.join(ckpt_dir, f"delta_epoch{e}.pth")
        torch.save(delta, ckpt_file)
        asr, ssim, lpips = eval(
            model = model, 
            testset = trainset, 
            delta = delta, 
            target_img = test_img, 
            fr_models = fr_models, 
            save_sample_img = False, 
            ckpt_dir = ckpt_dir, 
            calculate_sim = True, 
            config = config, 
            logger = logger
        )
        logger.log(f"epoch{e} trainset asr on id {trainset.identity}: {asr}", out=False)
        logger.log(f"trainset perceptual sim: ssim={ssim:.6f}, lpips={lpips:.6f}", out=False)
        logger.log('', out=False)
        peak_memory = torch.cuda.max_memory_allocated()
        logger.log(f"Peak memory usage: {peak_memory / (1024 ** 2):.2f} MB\n", out=False)

    return delta


@torch.no_grad()
def eval(
    model: DiffusionPipeline, 
    testset: Union[IDLatentPromptDataset, GroupLatentPromptDataset], 
    delta: torch.Tensor, 
    target_img: Image.Image,
    fr_models: ModelEnsemble,
    save_sample_img: bool,
    ckpt_dir: str,
    calculate_sim: bool,
    config, 
    logger: DatetimeLogger,
) -> Union[dict, Tuple[dict, float, float]]:
    """
    Tests the supplied delta on the testset and fr_models in config.training.test_model_names. Returns a dict in the form of {test_model_name: asr_on_model}
    """
    device = model.device
    res = config.dataset.res
    num_inference_steps = config.diffusion.diffusion_steps
    start_step = config.diffusion.start_step
    guidance_scale = config.diffusion.guidance_scale
    do_classifier_free_guidance = (guidance_scale > 0.0)
    test_model_names = config.training.test_model_names

    # generate test adv images
    eval_batch_size = 1
    testloader = DataLoader(testset, batch_size=eval_batch_size, shuffle=False)
    adv_images = []
    clean_images = []
    for batch in testloader:
        batch_diff_latents: torch.Tensor
        batch_images, batch_diff_latents, batch_prompt = batch
        batch_images = batch_images.to(device)
        clean_images.append(batch_images)
        batch_diff_latents = batch_diff_latents.to(device)
        batch_adv_latents = batch_diff_latents + delta
        batch_adv_images = sample(model, batch_prompt, batch_adv_latents, guidance_scale, num_inference_steps, start_step, return_tensor=True, res=res)
        adv_images.append(batch_adv_images)
    
    adv_images = torch.cat(adv_images, dim=0) # [-1.0, 1.0]
    clean_images = torch.cat(clean_images, dim=0) # [-1.0, 1.0]
    target_img = image2tensor(target_img, device)
    target_embedding = fr_models(target_img, test_model_names)

    if calculate_sim:
        ssim = calculate_ssim((clean_images + 1) / 2, (adv_images + 1) / 2)
        ssim = torch.mean(ssim).item()
        lpips = calculate_lpips(clean_images, adv_images)
        lpips = torch.mean(lpips).item()

    asc = {test_model_name:0 for test_model_name in test_model_names} # attack success count
    success_idx = []

    # test attack result
    for i in range(len(testset)):
        adv_image = adv_images[i][None]
        adv_embedding = fr_models(adv_image, test_model_names)

        for test_model_name in test_model_names:
            thres = THRES_DICT[test_model_name][1] # we use the 99 percent confidence threshold
            simi = F.cosine_similarity(target_embedding[test_model_name], adv_embedding[test_model_name])
            if simi > thres:
                asc[test_model_name] += 1
                success_idx.append(i)
    
    # store sample img
    if save_sample_img:
        if not success_idx:
            logger.log("No success images.")
            idx = 0
        else:
            idx = success_idx[0]
        sample_filename = testset.image_filenames[idx]
        if '_' in testset.identity: # num_id_in_group > 1
            testset_ids = testset.identity.split('_')
            img_id = testset_ids[idx // testset.num_from_each]
        else:
            img_id = testset.identity
        source_image: Image.Image = Image.open(os.path.join(config.dataset.images_root, img_id, sample_filename))
        sample_image = adv_images[idx][None]
        sample_image = tensor2image(sample_image)[0]
        sample_file_path = os.path.join(ckpt_dir, f"adv{img_id}_{sample_filename}")
        sample_image.save(sample_file_path)
        source_image = source_image.resize(sample_image.size)
        source_image.save(os.path.join(ckpt_dir, f"src{img_id}_{sample_filename}"))
        logger.log(f"Saved sample image adv{img_id}_{sample_filename}")
    
    asr = {k: v/len(testset) for k, v in asc.items()}

    if calculate_sim:
        return asr, ssim, lpips
    return asr

@torch.enable_grad()
def attack(
    model: DiffusionPipeline,
    trainsets: Union[List[IDLatentPromptDataset], List[GroupLatentPromptDataset]],
    testsets: Union[List[IDLatentPromptDataset], List[GroupLatentPromptDataset]],
    target_data: dict,
    fr_models: ModelEnsemble = None,
    config=None,
    logger: DatetimeLogger=None,
):
    # device = model.device
    res = config.dataset.res
    # num_inference_steps = config.diffusion.diffusion_steps
    # start_step = config.diffusion.start_step
    # guidance_scale = config.diffusion.guidance_scale
    # do_classifier_free_guidance = (guidance_scale > 0.0)

    train_timer = Timer()

    ids = []
    id_asrs = []
    id_ssims = []
    id_lpipss = []
    test_img = Image.open(config.dataset.test_path).resize((res, res))
    test_img = image2tensor(test_img, model.device)
    for id_i in range(len(trainsets)):
        train_timer.tic()

        trainset = trainsets[id_i]
        testset = testsets[id_i]

        ckpt_dir = os.path.join(logger.log_dir, f"{trainset.identity}")
        os.makedirs(ckpt_dir, exist_ok=True)

        logger.log(f"Victim id no {id_i}: {trainset.identity}")
        
        if config.training.memory_efficient:
            print("Using memory efficient training loop")
            delta = identity_attack_memory_efficient( # enter training loop
                model=model, 
                trainset=trainset, 
                target_data=target_data, 
                test_img=test_img,
                fr_models=fr_models, 
                ckpt_dir=ckpt_dir, 
                config=config, 
                logger=logger
            )
        else:
            print("Using regular training loop")
            delta = identity_attack_regular( # enter training loop
                model=model, 
                trainset=trainset, 
                target_data=target_data, 
                test_img=test_img,
                fr_models=fr_models, 
                ckpt_dir=ckpt_dir, 
                config=config, 
                logger=logger
            )

        asr, ssim, lpips = eval(
            model = model, 
            testset = testset, 
            delta = delta, 
            target_img = test_img, 
            fr_models = fr_models, 
            save_sample_img = True, 
            ckpt_dir = ckpt_dir, 
            calculate_sim = True, 
            config = config, 
            logger = logger    
        )
        peak_memory = torch.cuda.max_memory_allocated()
        logger.log(f"Peak memory usage: {peak_memory / (1024 ** 2):.2f} MB")

        ids.append(trainset.identity)
        id_asrs.append(asr)
        id_ssims.append(ssim)
        id_lpipss.append(lpips)

        iter_time = train_timer.toc(hms=False)
        average_time = train_timer.average(hms=False)
        iters_left = len(trainsets) - id_i - 1

        logger.log(f"testset asr on id {trainset.identity}: {asr}")
        logger.log(f"testset perceptual sim: ssim={ssim:.6f}, lpips={lpips:.6f}")
        logger.log(f"id training time: {sec2hms(iter_time)}. estimated remaining time: {sec2hms(average_time * iters_left)}")
        logger.log('\n')
        if (id_i + 1) % 10 == 0 and (id_i + 1) < len(trainsets):
            mean_asr = get_mean_asr(id_asrs)
            logger.log(f"mean metrics of {id_i + 1} ids:")
            logger.log(f"mean asr: {mean_asr}")
            logger.log(f"mean perceptual sim: ssim={sum(id_ssims)/len(ids):.6f}, lpips={sum(id_lpipss)/len(ids):.6f}")
            logger.log('\n\n')

    
    for id_i in range(len(ids)):
        logger.log(f"testset asr on id {ids[id_i]}: {id_asrs[id_i]}")
        logger.log(f"testset perceptual sim: ssim={id_ssims[id_i]:.6f}, lpips={id_lpipss[id_i]:.6f}")
        logger.log()
    
    mean_asr = get_mean_asr(id_asrs)
    logger.log(f"mean asr: {mean_asr}")
    logger.log(f"mean perceptual sim: ssim={sum(id_ssims)/len(ids):.6f}, lpips={sum(id_lpipss)/len(ids):.6f}")

                            
if __name__ == "__main__":
    pass